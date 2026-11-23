// Instruction grounding: natural language (Bahasa Indonesia or English) ->
// a symbolic plan over the objects the vision module can see.
// This is a small rule-based grounder (keywords + slots), NOT a large
// language model — it is deterministic, offline and runs on any phone. The
// server-side /act endpoint is where a large VLA would plug in instead.

export const COLORS = {
  red: ["red", "merah"], green: ["green", "hijau"], blue: ["blue", "biru"],
  yellow: ["yellow", "kuning"], purple: ["purple", "ungu"],
  orange: ["orange", "oranye", "jingga"], cyan: ["cyan", "tosca", "toska", "teal"],
};
// shape nouns are accepted but grounding is by color (one object per color)
export const SHAPES = { cube: ["cube", "kubus", "block", "balok"], cylinder: ["cylinder", "tabung", "can", "kaleng", "silinder"] };
const VERBS = {
  pick: ["pick", "grab", "take", "lift", "ambil", "angkat", "pegang"],
  place: ["place", "put", "drop", "move", "taruh", "letakkan", "pindahkan", "simpan", "bawa"],
  reach: ["reach", "touch", "point", "go", "sentuh", "tunjuk", "dekati", "ke"],
  home: ["home", "reset", "rest", "pulang", "kembali", "istirahat"],
  open: ["open", "release", "buka", "lepas"],
  close: ["close", "grip", "tutup", "genggam"],
  stack: ["stack", "tumpuk"],
};
const words = (s) => s.toLowerCase().normalize("NFKD").replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter(Boolean);
const has = (ws, list) => ws.some((w) => list.includes(w));

function colorsIn(ws) {
  const out = [];
  ws.forEach((w, i) => { for (const [c, keys] of Object.entries(COLORS)) if (keys.includes(w)) out.push({ c, i }); });
  return out;
}

/**
 * @param {string} text
 * @param {{cubes: string[], pads: string[]}} scene names of visible cubes/pads
 * @returns {{ok: boolean, plan?: object[], error?: string, parse: object}}
 */
export function ground(text, scene) {
  const ws = words(text);
  const cols = colorsIn(ws);
  const parse = { tokens: ws, colors: cols.map((x) => x.c) };
  if (!ws.length) return { ok: false, error: "instruksi kosong", parse };
  if (has(ws, VERBS.home)) return { ok: true, plan: [{ skill: "home" }], parse };
  const cubeCols = cols.filter((x) => scene.cubes.includes(x.c));
  const padCols = cols.filter((x) => scene.pads.includes(x.c));
  const isPlace = has(ws, VERBS.place) || has(ws, VERBS.stack) || ws.includes("on") || ws.includes("di") || ws.includes("atas");
  if ((has(ws, VERBS.pick) || isPlace) && cubeCols.length) {
    const obj = cubeCols[0].c;
    // destination = a pad, or a second cube (stacking), mentioned AFTER the object
    const dest = [...padCols, ...cubeCols.slice(1)].filter((x) => x.i > cubeCols[0].i).sort((a, b) => a.i - b.i)[0];
    const plan = [{ skill: "pick", object: obj }];
    if (dest) plan.push({ skill: "place", target: dest.c, onCube: scene.cubes.includes(dest.c) });
    else if (isPlace && !has(ws, VERBS.pick)) return { ok: false, error: "tujuan tidak disebut (mis. 'ke pad kuning')", parse };
    return { ok: true, plan, parse };
  }
  if (has(ws, VERBS.reach) && cols.length) return { ok: true, plan: [{ skill: "reach", target: cols[0].c }], parse };
  if (has(ws, VERBS.open)) return { ok: true, plan: [{ skill: "gripper", open: true }], parse };
  if (has(ws, VERBS.close)) return { ok: true, plan: [{ skill: "gripper", open: false }], parse };
  const missing = cols.filter((x) => !scene.cubes.includes(x.c) && !scene.pads.includes(x.c));
  if (missing.length) {
    return { ok: false, error: `objek ${missing.map((x) => x.c).join(", ")} tidak ada/terlihat di meja — yang terlihat: ${[...scene.cubes, ...scene.pads].join(", ") || "tidak ada"}`, parse };
  }
  return { ok: false, error: "instruksi tidak dipahami — coba: 'ambil kubus merah taruh di pad kuning'", parse };
}
