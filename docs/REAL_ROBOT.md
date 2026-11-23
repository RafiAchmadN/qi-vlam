# Data Robot Nyata → Model → Kompresi → Robot Fisik

Lapisan ini melatih policy dari **demonstrasi robot fisik** (bukan simulasi),
membandingkan beberapa arsitektur, mengompresinya, lalu menjalankannya:
(1) di aplikasi, langsung di perangkat, pada frame kamera nyata yang diputar
ulang; (2) di **robot SO-100/SO-101 fisik** lewat paket `lerobot`.

## 1. Data nyata (LeRobot hub, terbuka)

| Dataset | Robot fisik | Tugas | Episode | Frame @10 Hz |
|---|---|---|---|---|
| `lerobot/svla_so101_pickplace` | SO-101 | lego merah muda ke kotak bening | 50 | 3.995 |
| `lerobot/svla_so100_pickplace` | SO-100 | ambil kubus, taruh di kotak | 50 | 6.559 |
| `lerobot/svla_so100_stacking` | SO-100 | kubus merah di atas kubus biru | 56 | 7.667 |
| `lerobot/svla_so100_sorting` | SO-100 | kubus merah ke kotak kanan, biru ke kiri | 52 | 11.918 |
| `lerobot/koch_pick_place_5_lego` | Koch v1.1 | lego ke kotak | 50 | 12.672 |

Semuanya demonstrasi teleoperasi (leader–follower) pada lengan sungguhan,
30 fps, dengan dua kamera (video AV1) dan posisi 6 sendi. Satuan sendi
mengikuti kalibrasi LeRobot masing-masing dataset: SO-101 dinormalisasi
−100…100 (gripper 0…100), sedangkan SO-100 dan Koch berupa derajat. Karena
itu setiap dataset dinormalisasi dengan statistiknya sendiri.

```bash
python -m src.real.download            # parquet + kamera utama SO-100/101 (~1,4 GB → data/real, gitignored)
python -m src.real.download --datasets koch_pick_place_5_lego --cameras observation.images.phone
python -m src.real.data                # decode AV1 (PyAV/libdav1d) → cache 64×64 @10 Hz
```

## 2. Model yang dibandingkan (semua memprediksi *action chunk* 10 langkah = 1 detik)

| Model | Isi |
|---|---|
| `state_mlp` | sendi + tugas saja, **tanpa kamera** (kontrol: apakah kamera dibutuhkan?) |
| `cnn_mlp` | encoder konvolusi + keypoint spatial-softmax + MLP (policy visuomotor utama) |
| `act_tf` | token fitur kamera + token sendi/tugas → Transformer 2 lapis (ACT-lite) |
| `healing_ctrl` | `cnn_mlp` + langkah latih tambahan yang sama seperti healing (kontrol yang adil) |
| `cnn_ttns` / `cnn_mps` | head MLP dikompresi tensor network χ=16 + healing (quantum-inspired) |
| `cnn_ttns_noheal` | kompresi TTNS tanpa healing |
| `cnn_int8` | kuantisasi INT8 per kanal pasca-training (pembanding klasik) |
| `head_tanh` / `head_surrogate` / `head_pqc` | trunk dibekukan → bottleneck 6 unit: tanh, surrogate kosinus, PQC 6 qubit (PennyLane) |

Augmentasi random-shift (DrQ) dipakai untuk gambar. Split **per episode**
(75/10/15), 3 seed, dan 6.000 langkah AdamW.

```bash
python -m src.real.train_real --seeds 3 --steps 6000     # → results/real_benchmark.json, models/real/*.pt
```

### Metrik (open-loop, episode uji yang tidak dipakai training)

* `motion_nmse`: galat **perpindahan** yang diprediksi relatif terhadap pose
  saat ini, sepanjang chunk. Nilai 1,0 = "diam saja". Ini metrik paling jujur,
  karena nMSE aksi-berikutnya didominasi autokorelasi pose (di 10 Hz, aksi
  berikutnya hampir sama dengan pose sekarang), sehingga model tanpa kamera
  pun terlihat bagus.
* `nmse_first`, `nmse_chunk`: galat aksi yang dinormalisasi.
* `joint_mae`: galat absolut 5 sendi lengan dalam satuan dataset.
* `grip_acc`: kecocokan status gripper (buka/tutup).
* `params`, `storage_kb`, `latency_ms` (CPU 1 thread, batch 1).

Hasilnya ada di `results/real_benchmark.json` dan laporan §6.10. Ringkasan
untuk SO-100/SO-101 (3 seed, 208 episode, episode uji):

| Model | Param | Ukuran | Galat gerakan ↓ | MAE sendi | Gripper | CPU |
|---|---|---|---|---|---|---|
| kontrol healing | 379.836 | 1.484 KB | **0,249 ± 0,008** | 1,56 | 96,9% | 2,7 ms |
| CNN dense | 379.836 | 1.484 KB | 0,255 ± 0,007 | 1,60 | 96,9% | 2,9 ms |
| CNN INT8 | 379.836 | **807 KB** | 0,255 ± 0,007 | 1,60 | 96,9% | 3,0 ms |
| ACT Transformer | 520.636 | 2.034 KB | 0,256 ± 0,027 | 1,56 | 97,0% | 3,9 ms |
| **CNN + TTNS + healing** | **175.452** | **685 KB** | 0,268 ± 0,006 | 1,82 | 96,8% | 2,9 ms |
| tanpa kamera | 149.820 | 585 KB | 0,351 ± 0,012 | 1,44 | 96,6% | 0,3 ms |
| CNN + MPS + healing | 209.532 | 818 KB | 0,374 ± 0,008 | 2,28 | 95,7% | 4,5 ms |
| head tanh 6 unit | 381.798 | 1.491 KB | 0,754 ± 0,037 | 7,61 | 88,8% | 2,7 ms |
| TTNS tanpa healing | 175.452 | 685 KB | 0,771 ± 0,041 | 5,13 | 95,5% | 2,2 ms |
| head PQC kuantum | 381.840 | 1.492 KB | 1,245 ± 0,041 | 10,37 | 85,3% | 15,9 ms |
| head surrogate | 381.828 | 1.492 KB | 1,267 ± 0,145 | 9,92 | 85,4% | 3,1 ms |

Kesimpulan:
* kamera menurunkan galat gerakan 27%
* INT8 (−46% ukuran) dan TTNS (2,2× lebih sedikit parameter) tetap layak dipakai
* MPS kalah dari TTNS
* healing wajib
* head kuantum tidak memberi keunggulan

### Robot kedua: Koch v1.1 (data nyata, dilatih terpisah)

`lerobot/koch_pick_place_5_lego`, 50 episode, 12.672 frame, kamera `phone`.
Satuan sendi Koch berbeda dari SO-100/101, jadi model dilatih terpisah
(bukan digabung), 3 seed, episode uji yang sama untuk semua model.

```bash
python -m src.real.train_real --datasets koch_pick_place_5_lego --seeds 3 --steps 6000 --heal-steps 1200 \
    --models state_mlp,cnn_mlp,act_tf,healing_ctrl,cnn_ttns,cnn_ttns_noheal,cnn_int8 \
    --out results/real_benchmark_koch.json --models-dir models/real_koch
```

| Model | Param | Ukuran | Galat gerakan ↓ | MAE sendi | Gripper | CPU |
|---|---|---|---|---|---|---|
| ACT Transformer | 520.252 | 2.032 KB | **0,088 ± 0,006** | 2,25 | 97,9% | 3,9 ms |
| CNN dense | 379.068 | 1.481 KB | 0,091 ± 0,010 | 2,63 | 97,9% | 2,5 ms |
| kontrol healing | 379.068 | 1.481 KB | 0,091 ± 0,010 | 2,56 | 97,9% | 2,6 ms |
| CNN INT8 | 379.068 | **806 KB** | 0,091 ± 0,010 | 2,64 | 97,9% | 2,5 ms |
| **CNN + TTNS + healing** | **175.404** | **685 KB** | 0,094 ± 0,009 | 3,04 | 97,8% | 2,7 ms |
| CNN + MPS + healing | 194.892 | 761 KB | 0,110 ± 0,004 | 3,54 | 97,7% | 4,3 ms |
| tanpa kamera | 149.052 | 582 KB | 0,150 ± 0,002 | 1,92 | 98,3% | 0,2 ms |
| TTNS tanpa healing | 175.404 | 685 KB | 0,255 ± 0,037 | 8,93 | 97,0% | 2,1 ms |
| head tanh 6 unit | 381.030 | 1.488 KB | 0,267 ± 0,014 | 11,47 | 94,4% | 2,9 ms |
| head PQC kuantum | 381.072 | 1.489 KB | 0,356 ± 0,074 | 13,43 | 93,5% | 16,5 ms |
| head surrogate | 381.060 | 1.489 KB | 0,389 ± 0,029 | 13,41 | 93,1% | 3,0 ms |

Pola di robot kedua sama dengan SO-100/101:
* kamera menurunkan galat gerakan 39% (0,150 → 0,091)
* TTNS 2,2× lebih kecil hanya +3% galat (0,094 vs 0,091), dalam rentang std antar-seed
* tanpa healing galatnya 2,8× lebih besar
* ACT sedikit unggul, tetapi selisihnya lebih kecil dari std
* MPS kalah dari TTNS lagi (0,110 vs 0,094) walau parameternya lebih banyak
* head PQC: lebih baik dari surrogate kosinus (0,356 vs 0,389) tetapi kalah dari
  tanh klasik (0,267), variansnya terbesar (0,26–0,44 antar seed) dan 5,6× lebih lambat.
  Di SO-100/101 PQC juga kalah dari tanh. Tidak ada keunggulan kuantum.

Tugas Koch lebih mudah dari campuran 4 tugas SO-100/101 (galat absolut 0,09 vs
0,25), jadi angka antar-robot tidak dibandingkan langsung; yang dibandingkan
adalah urutan model di dalam tiap robot.

MPS dan head bottleneck dilatih pada run kedua (`--models cnn_mlp,cnn_mps,head_tanh,head_surrogate,head_pqc
--out results/real_benchmark_koch_extra.json`) lalu digabung dengan
`python scripts/merge_real_results.py results/real_benchmark_koch.json results/real_benchmark_koch_extra.json`.
Base CNN run kedua identik bit-per-bit dengan run pertama di ketiga seed
(`reproducibility` di JSON), jadi kedua run memakai model dasar dan episode uji yang sama.

## 3. Model berjalan di aplikasi (di HP), pada data nyata

Pilih **Data robot nyata (SO-100/101)** di header. Aplikasi memutar episode uji
(frame kamera nyata 64×64 dan sendi terukur). Setiap 100 ms, model vision
(CNN + head TTNS terkompresi) **dijalankan di perangkat** dengan runtime JS
murni (`app/src/real/visionPolicy.js`, paritas dengan PyTorch ~5e-6, diuji),
lengkap dengan temporal ensembling seperti ACT. Grafik menampilkan prediksi
model di samping perintah operator manusia, beserta MAE, kecocokan gripper,
dan latensi inferensi di perangkat.

```bash
python -m src.real.export_web --model models/real/cnn_ttns.pt   # → app/public/models/real_cnn_ttns.json + app/public/real/*
```

## 4. Menjalankan di robot fisik

```bash
python -m src.real.deploy replay --model models/real/cnn_ttns.pt --dataset svla_so101_pickplace
#   loop kontrol penuh pada episode nyata: latensi per keputusan & cek real-time

pip install lerobot                       # driver bus motor Feetech + kamera
python -m src.real.deploy robot --model models/real/cnn_ttns.pt --dataset svla_so101_pickplace \
    --robot so101 --port COM5 --camera 0 --cam-name up --dry-run     # baca & cetak, TIDAK mengirim
python -m src.real.deploy robot ... --max-delta 3 --seconds 20       # lalu jalankan pelan
```

Syarat agar model dataset X bisa dipakai di robot Anda:

1. **Robot dan kalibrasi sama dengan dataset**: SO-101 dengan kalibrasi
   LeRobot untuk `svla_so101_*`, dan seterusnya. Kalau satuannya berbeda,
   aksinya akan salah.
2. **Kamera di posisi yang mirip**: `up` = dari atas meja untuk SO-101
   pickplace, `top` untuk SO-100. Sudut pandang lain berada di luar
   distribusi training.
3. **Objek dan tugas yang sama**: lego merah muda dan kotak bening, dan
   seterusnya. Model hanya meniru tugas yang ada di datanya.

Pengaman di skrip:
* aksi di-clamp ke rentang min/maks yang pernah tercatat di data
* perubahan per langkah dibatasi `--max-delta`
* `--dry-run` tidak mengirim perintah apa pun
* Ctrl-C menahan pose terukur

Batas torsi dan limit servo tetap berlaku di firmware.

**Status jujur:** jalur fisik ini belum diuji di hardware di repo ini (tidak
ada robot di mesin pengembang). Yang sudah diverifikasi:
* training dan evaluasi pada data robot nyata
* replay loop kontrol pada episode nyata dengan latensi real-time
* paritas runtime JS ↔ PyTorch
* logika pengaman (unit test)

Sukses tugas closed-loop baru bisa diukur di robot fisik. Untuk itu, mulai
dengan `--dry-run`, lalu `--max-delta` kecil, dengan tangan siap di catu daya.
