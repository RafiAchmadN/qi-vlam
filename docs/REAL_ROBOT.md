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
