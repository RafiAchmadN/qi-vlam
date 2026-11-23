#!/usr/bin/env python3
"""Partial MPS compression of REAL LLaVA-1.5-7B — Colab/Kaggle GPU script.

Run this on Google Colab (T4, free) or Kaggle (P100/T4). Produces
llava_results.json with per-layer compression stats and perplexity
before/after on a small held-out text set — evidence on a real
7B vision-language model, complementing the CPU proxy benchmark.

Colab cell 1:
    !pip install -q transformers accelerate bitsandbytes
    !git clone <your-repo> && cd <your-repo>
Colab cell 2:
    !python scripts/compress_llava_colab.py --layers 4 --bond-dimension 384 \
        --method ttns --heal-steps 25 --heal-lr 5e-6

Memory strategy: load in fp16 with device_map='auto'; compress only the MLP
(gate/up/down) projections of the first N decoder layers. This keeps the run
inside free-tier VRAM while measuring real 7B weight-spectrum behavior.
model.requires_grad_(False) is called once up front (from_pretrained leaves
every parameter trainable by default) and gradient checkpointing is enabled
— without both, loss.backward() during healing computes/stores gradients
and activations for the entire ~7B backbone, not just the tiny compressed
cores, which reliably exceeds a T4's 15GB VRAM (hit for real on Colab: a
plain CUDA/backward failure with the backbone left trainable).

Compression + healing run SEQUENTIALLY, one layer at a time (compress layer
i, heal layer i immediately, only then compress layer i+1) — GPTQ/AWQ-style
calibration, not compress-everything-then-heal-everything. This was found
necessary empirically: at bond_dimension=768 (rel_err ~0.6-0.77/layer,
individually fine) compressing 4 layers before any healing produced a
forward-pass NaN (fp16 overflow through the residual stream) before healing
could even start; healing each layer right after compressing it means every
later layer sees an already-partially-corrected signal, not raw compounding
error. --heal-steps applies PER LAYER now (total optimizer steps = layers *
heal-steps), not once for the whole model.

That fix alone was NOT sufficient — run for real on Colab with
--heal-steps 100 at bond_dimension=768: layer0 healed "successfully" (loss
3.13 -> 0.28) but perplexity_after_layer0 dropped from 20.6 to 1.3, an
implausibly large improvement caused by HEAL_TEXTS/EVAL_TEXTS being the
SAME 2-sentence set at the time (train-on-eval-set leak, since fixed by
splitting them below) — 100 steps on 2 sentences is enough to memorize them
outright. That overfit drift in layer0's cores was then large enough that
layer1's healing diverged to NaN/Inf on its very first optimizer step
(diverged_at_step=0, empty loss history), cascading NaN through layer2 and
layer3. Lower --heal-steps (avoid overfitting an early layer this hard),
lower --heal-lr, and a lower --bond-dimension (smaller initial rel_err
feeding into the fragile healing dynamics) are the mitigations tried next —
none yet verified on real Colab GPU as of this writing.

Healing fine-tunes MPS/TTNS cores only (ordinary autograd, HF's built-in
labels= cross-entropy loss, fp32 via FP32Island) on HEAL_TEXTS below, a
small built-in smoke-test set — NOT the real nuScenes-QA / LLaVA-Instruct
subset. perplexity() is measured on the disjoint EVAL_TEXTS set (see the
train-on-eval-set bug above for why these must not overlap). Swap both for
a real dataset loader before citing perplexity_final in a report (see
CLAUDE.md honest-claims policy) — the loop itself is real and working,
only the data source is currently a stand-in.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn


class FP32Island(nn.Module):
    """Wraps a compressed layer so it trains in fp32 while embedded in an
    fp16 model — proper mixed-precision "master weights" for the trainable
    MPS/TTNS cores, casting only at the activation boundary.

    Why this exists: the whole LLaVA backbone loads as fp16
    (`torch_dtype=torch.float16`, needed to fit a 7B model in T4 VRAM). If
    the compressed cores are ALSO fp16 and trained directly with AdamW, the
    optimizer's moment estimates and parameter storage inherit fp16's ~3
    decimal digits of precision and narrow-ish dynamic range — on a model
    already pushed to ~90%+ relative error by aggressive compression (so
    starting loss is already near-random, ln(vocab)-ish), this reliably
    diverges to NaN within 1-2 steps even with a low lr and gradient
    clipping (both observed empirically running this script for real on
    Colab). Keeping just the small compressed-core parameters in fp32 fixes
    this at negligible memory cost (that's the whole point of compressing
    them in the first place) without touching the frozen fp16 backbone.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner  # kept fp32, never .half()'d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x.float()).to(x.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--layers", type=int, default=4,
                    help="number of decoder layers to compress")
    ap.add_argument("--bond-dimension", type=int, default=96)
    ap.add_argument("--method", choices=["mps", "ttns"], default="mps",
                    help="mps = 3-core TT-SVD; ttns = single truncated SVD. "
                         "On the CPU proxy tower (pareto.json) ttns is far "
                         "gentler than mps at matched bond dimension (chi=8: "
                         "52%% drop vs 4.5%%) — but run on real LLaVA-7B "
                         "weights at chi=96, BOTH gave rel_err ~0.88-0.96 "
                         "(near-total information loss), so the method choice "
                         "is NOT the bottleneck here: real 7B MLP weights are "
                         "far higher effective rank than the synthetic "
                         "rank-16 proxy task. --bond-dimension needs to be "
                         "raised substantially (try a 1-layer sweep, e.g. "
                         "--layers 1, at chi in {192, 384, 768} before "
                         "committing to a multi-layer run) rather than "
                         "switching method.")
    ap.add_argument("--eval-samples", type=int, default=32)
    ap.add_argument("--heal-steps", type=int, default=0,
                    help="0 = skip healing (default); >0 fine-tunes MPS cores "
                         "only for this many steps on the built-in smoke-test "
                         "text set (NOT nuScenes-QA/LLaVA-Instruct — swap "
                         "`texts` for a real dataset loader before trusting "
                         "these numbers for a report)")
    ap.add_argument("--heal-lr", type=float, default=1e-5,
                    help="fp16 healing on an already-degraded model is prone "
                         "to NaN blowup at higher lr (observed at 1e-4); "
                         "combined with --heal-grad-clip, lower is safer")
    ap.add_argument("--heal-grad-clip", type=float, default=1.0)
    ap.add_argument("--out", default="llava_results.json")
    args = ap.parse_args()

    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from src.compression.tensor_network import MPSLinear, TTNSLinear

    print("Loading model (fp16)...")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto",
        low_cpu_mem_usage=True)
    processor = AutoProcessor.from_pretrained(args.model)
    # transformers >=4.5x refactored LlavaForConditionalGeneration to wrap
    # everything in a LlavaModel at .model (vision_tower, multi_modal_projector,
    # language_model), with lm_head/loss living on the outer
    # LlavaForConditionalGeneration itself — verified against the installed
    # transformers source (modeling_llava.py), not assumed. `model.language_model`
    # does not exist directly; `model.model.language_model` is the decoder
    # stack (AutoModel.from_config, e.g. LlamaModel) and already has `.layers`
    # without an extra `.model` hop.
    lm = model.model.language_model

    # from_pretrained leaves every parameter requires_grad=True by default —
    # without this, loss.backward() during healing (below) computes and
    # stores gradients for the ENTIRE ~7B-parameter backbone, not just the
    # small compressed cores unfrozen per-layer just before healing. On a
    # 15GB T4 with the model already ~14GB in fp16 (some layers offloaded to
    # CPU by device_map="auto"), that reliably blows the VRAM/backward
    # budget. Freeze everything once up front; the per-layer loop below only
    # ever re-enables grad on its own small compressed-core parameters.
    model.requires_grad_(False)

    # Even with the backbone frozen, healing an early layer (e.g. layer0)
    # still needs the loss to backprop through every subsequent frozen
    # layer's activations to reach that layer's compressed cores — that
    # activation memory alone can still exceed a T4's 15GB. Gradient
    # checkpointing trades compute (recomputes each layer's forward during
    # backward) for activation memory; use_cache=False avoids the standard
    # HF warning about the two being redundant during training.
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # --- perplexity eval helper on plain text (proxy for VQA quality drift)
    # HEAL_TEXTS and EVAL_TEXTS are disjoint on purpose: an earlier run healed
    # and measured perplexity on the exact same 2-sentence set, so 100 heal
    # steps drove the loss to near-zero by memorizing those 2 sentences
    # (perplexity_after_layer0 dropped from 20.6 to 1.3) rather than showing
    # any real generalization — a textbook train-on-eval-set leak. Splitting
    # them doesn't make this a real benchmark (still a synthetic smoke-test
    # pool, not nuScenes-QA/LLaVA-Instruct — see module docstring), but it
    # at least stops perplexity_final from measuring memorization.
    HEAL_TEXTS = [
        "The autonomous vehicle detected a pedestrian crossing the street.",
        "A robot arm picks up the red block and places it on the shelf.",
        "The car slows down as it approaches a red traffic light.",
        "The gripper closes around the object before lifting it.",
        "A cyclist is riding along the right side of the road.",
        "The vision system identifies a stop sign ahead of the vehicle.",
        "The robotic arm rotates its wrist to align with the target.",
        "Two pedestrians are waiting at the crosswalk for the signal.",
        "The mobile base navigates around an obstacle in its path.",
        "The manipulator releases the object into the bin.",
        "A truck merges into the lane ahead of the autonomous car.",
        "The sensor detects a lane marking curving to the left.",
    ]
    EVAL_TEXTS = [
        "The camera captures a clear view of the intersection.",
        "The robot's end effector approaches the target slowly.",
        "The vehicle maintains a safe following distance from traffic.",
        "The arm retracts after completing the placement task.",
        "A dog runs across the street in front of the car.",
        "The gripper applies just enough force to hold the fragile item.",
        "The car signals before changing into the adjacent lane.",
        "The robot pauses briefly to re-plan its trajectory.",
    ]
    texts = [EVAL_TEXTS[i % len(EVAL_TEXTS)] for i in range(args.eval_samples)]

    @torch.no_grad()
    def perplexity():
        tok = processor.tokenizer
        losses = []
        for t in texts:
            ids = tok(t, return_tensors="pt").input_ids.to(model.device)
            # loss_function/lm_head live on the outer ForConditionalGeneration,
            # not on the bare language_model backbone (see note above) —
            # pixel_values is Optional[...] = None in its forward signature,
            # so a text-only call is valid.
            out = model(input_ids=ids, labels=ids)
            losses.append(out.loss.float().item())
        return float(torch.tensor(losses).mean().exp())

    ppl_before = perplexity()
    print(f"Perplexity before: {ppl_before:.3f}")

    # --- SEQUENTIAL per-layer compress-THEN-heal (GPTQ/AWQ-style) ---
    # Earlier version compressed ALL layers first, healed ALL together after.
    # Run for real on Colab: at bond_dimension=768 (rel_err ~0.6-0.77/layer,
    # each layer individually fine), compressing 4 layers simultaneously
    # produced NaN in the forward pass BEFORE healing could even start — each
    # layer's error compounds through the residual stream faster than a
    # single post-hoc healing pass can correct. Compressing+healing ONE layer
    # at a time means every later layer sees an already-partially-corrected
    # signal from earlier layers, not raw compounding error — same principle
    # as sequential calibration in GPTQ/AWQ. --heal-steps now applies PER
    # LAYER (total optimizer steps = layers * heal-steps), not once overall.
    layer_stats = {}
    heal_by_layer = {}
    ppl_trajectory = [{"stage": "before", "perplexity": ppl_before}]
    t0 = time.time()
    tok = processor.tokenizer
    for li in range(min(args.layers, len(lm.layers))):
        mlp = lm.layers[li].mlp
        layer_trainable = []
        for name in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(mlp, name).float().cpu()
            if args.method == "mps":
                compressed = MPSLinear.from_linear(lin, bond_dimension=args.bond_dimension,
                                                   num_cores=3)
                ratio, rel_err = compressed.stats.compression_ratio, compressed.stats.relative_error
            else:  # ttns: single truncated SVD, no chained-core error compounding
                compressed = TTNSLinear.from_linear(lin, bond_dimension=args.bond_dimension)
                W = lin.weight.data
                rel_err = (torch.norm(W - compressed.reconstruct_weight()) / torch.norm(W)).item()
                ratio = compressed.get_compression_ratio()
            # Kept fp32 (see FP32Island docstring) — not .half()'d.
            compressed = FP32Island(compressed.to(model.device))
            setattr(mlp, name, compressed)
            layer_stats[f"layer{li}.{name}"] = {"ratio": ratio, "rel_err": rel_err}
            print(f"  layer{li}.{name}: {ratio:.2f}x, rel_err {rel_err:.4f}")
            layer_trainable.extend(compressed.parameters())

        # --- heal THIS layer only, right now, before compressing the next ---
        # Not routed through VLAMTrainer (src/training/vlam_trainer.py): that
        # trainer's interface is batch_fn -> (x, y) + nn.MSELoss, built for
        # the CPU proxy regression tower; LLaVA is fine-tuned via HF's
        # built-in labels= cross-entropy loss instead (see perplexity()) — a
        # different interface, same engineering call as keeping TTNS
        # honestly separate from MPS (CLAUDE.md invariant #7).
        layer_losses: list = []
        layer_diverged_at = None
        if args.heal_steps > 0:
            for p in layer_trainable:
                p.requires_grad_(True)
            opt = torch.optim.AdamW(layer_trainable, lr=args.heal_lr)
            for step in range(args.heal_steps):
                t = HEAL_TEXTS[step % len(HEAL_TEXTS)]
                ids = tok(t, return_tensors="pt").input_ids.to(model.device)
                out = model(input_ids=ids, labels=ids)
                loss = out.loss.float()
                if not torch.isfinite(loss):
                    layer_diverged_at = step
                    print(f"  layer{li} heal step {step}: loss=NaN/Inf — "
                         f"stopping this layer's healing early (rel_err "
                         "above still too high for --heal-lr, or "
                         "--bond-dimension needs raising for this layer).")
                    break
                opt.zero_grad()
                loss.backward()
                if args.heal_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(layer_trainable, args.heal_grad_clip)
                opt.step()
                layer_losses.append(float(loss.item()))
            for p in layer_trainable:
                p.requires_grad_(False)  # freeze again before the next layer
            if layer_losses:
                print(f"  layer{li} healed: {len(layer_losses)} steps, "
                     f"final_loss={layer_losses[-1]:.4f}")
        heal_by_layer[f"layer{li}"] = {"losses": layer_losses, "diverged_at_step": layer_diverged_at}

        ppl_li = perplexity()
        ppl_trajectory.append({"stage": f"after_layer{li}", "perplexity": ppl_li})
        print(f"  perplexity after layer{li}: {ppl_li:.3f}")
    compress_time = time.time() - t0

    ppl_final = ppl_trajectory[-1]["perplexity"]
    print(f"Perplexity final: {ppl_final:.3f}")

    results = {
        "model": args.model,
        "method": args.method,
        "bond_dimension": args.bond_dimension,
        "layers_compressed": args.layers,
        "sequential_compress_and_heal": True,
        "perplexity_before": ppl_before,
        "perplexity_final": ppl_final,
        "perplexity_trajectory": ppl_trajectory,
        "heal_steps_per_layer": args.heal_steps,
        "heal_lr": args.heal_lr,
        "heal_grad_clip": args.heal_grad_clip,
        "heal_by_layer": heal_by_layer,
        "compress_and_heal_wall_clock_s": compress_time,
        "layer_stats": layer_stats,
        "mean_ratio": sum(v["ratio"] for v in layer_stats.values()) / len(layer_stats),
        "note": ("Sequential per-layer compress-then-heal (see module docstring) — "
                 "fixes the compounding-error NaN seen when compressing multiple "
                 "layers before any healing. heal_by_layer/perplexity_trajectory "
                 "let you see exactly where (if anywhere) a layer's healing "
                 "diverged, rather than one pass/fail number for the whole run. "
                 "Healing uses the small built-in smoke-test text set below, NOT "
                 "a real nuScenes-QA/LLaVA-Instruct held-out split — swap `texts` "
                 "for a real dataset loader before citing perplexity_final in a "
                 "report; see CLAUDE.md honest-claims policy."),
    }
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
