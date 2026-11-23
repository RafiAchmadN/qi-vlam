# Simulasi, Lingkungan Training & Integrasi ROS

Dokumen ini menjelaskan lapisan simulasi QI-VLA: lingkungan berbasis tempat,
simulasi fisika yang meniru ketidaksempurnaan robot nyata, pembuatan data
training, pelatihan model baru, dan cara menyambungkannya ke aplikasi, ROS 2,
serta simulator lain.

```
app/public/envs/*.json  ──►  Aplikasi (Three.js)           ──►  demo terekam (qi-vla-demos/1)
   (5 tempat + DR)       ├─►  MuJoCo (src/sim/physics.py)  ──►  dataset gaya LeRobot (parquet)
                         ├─►  Gymnasium  QiVla/PickPlace-v0          │
                         ├─►  MJCF (sim/mjcf/*.xml)                  ▼
                         └─►  URDF → ROS 2 (ros2/qi_vla_ros)   src/sim/train_pickplace.py
                                                                     │  dense + TTNS
                                                                     ▼
                                                     app/public/models/pickplace_*.json
```

## 1. Lingkungan berbasis tempat

| id | Tempat | Rintangan | Dinamika nominal |
|---|---|---|---|
| `lab_table` | Meja lab | – | lag servo 60 ms |
| `kitchen_counter` | Dapur | backsplash, ketel | lag 80 ms, backlash 2 mrad |
| `warehouse_shelf` | Gudang | tiang rak, papan rak | lag 100 ms, latensi 1 langkah, servo lebih lemah |
| `classroom_desk` | Meja kelas | tumpukan buku, tempat pensil | lag 70 ms |
| `factory_conveyor` | Pabrik (**held-out**) | konveyor, pagar | servo aus: lag 160 ms, latensi 2, backlash 6 mrad |

Satu file JSON per tempat (`app/public/envs/`) berisi: ukuran/warna meja,
pencahayaan, rintangan (kotak), area spawn objek, daftar objek dan bentuknya,
serta parameter dinamika.

**Objek**: setiap scene berisi 3–5 objek yang dipilih acak dari 5 warna
(merah, hijau, biru, oranye, tosca), berbentuk kubus 30 mm atau tabung
r=15 mm, ditambah 2 pad (kuning, ungu). Tinggi semua objek sama (30 mm),
sehingga tinggi genggam dan fitur policy tidak bergantung pada bentuk.
Jumlah objek ikut diacak sebagai domain randomization tingkat kekacauan
meja. Detektor HSV memisahkan ketujuh warna, dan ini diuji. Instruksi
menyebut bentuk ("ambil tabung oranye…", "stack the cyan cylinder…"),
sedangkan grounding tetap berdasarkan warna.
`randomization.json` berisi rentang **domain randomization**: lag servo,
latensi, backlash, noise encoder, noise deteksi objek, kekuatan servo, massa
link/kubus, gesekan, ukuran kubus (fisika), serta cahaya, warna meja, dan
kalibrasi kamera (visual, di aplikasi).

Warna meja dan rintangan sengaja bersaturasi rendah (HSV s < 0,45) supaya
detektor warna tidak salah mendeteksinya sebagai objek.

## 2. Simulasi fisika (MuJoCo)

`src/sim/physics.py::ArmSim`. Lengan dibangun dari `src/robot/arm.py`
(`src/sim/mjcf.py`), sehingga posisi ujung gripper **identik** dengan `arm.fk`
(diuji sampai 1e-9). Hal-hal yang meniru dunia nyata:

* gravitasi, kontak, dan **genggaman murni gesekan**: tidak ada weld/attach,
  sehingga kubus bisa tergelincir atau jatuh
* model servo: latensi perintah, lag orde-1, backlash (dead-band
  histeresis), batas torsi, dan droop akibat gravitasi
* noise encoder sendi dan noise posisi objek, yang mewakili galat deteksi kamera
* deteksi tabrakan lengan dengan meja atau rintangan (penetrasi > 2 mm)

Expert berskrip (`src/sim/expert.py`) bersifat closed-loop pada sendi
*terukur*, dengan aksi integral dan anti-windup, menunggu gripper benar-benar
menutup, serta melakukan visual servoing untuk mengoreksi kubus yang
tergelincir di genggaman. Wrist roll diatur supaya jari menjepit dua sisi
kubus.

## 3. Data training

```bash
python -m src.sim.collect --out data/qivla_sim --episodes 1600                      # dengan DR
python -m src.sim.collect --out data/qivla_sim_nodr --episodes 1600 --no-randomize --seed 100000
python -m src.sim.collect --images      # + gambar kamera atas 160x128 (PNG di parquet)
```

Formatnya mengikuti **tata letak LeRobot v3.0**: `data/chunk-000/*.parquet`,
`meta/info.json`, `meta/tasks.parquet`, `meta/episodes/…`, dan `meta/stats.json`.
`observation.state` berisi posisi sendi, `action` berisi target sendi absolut
(konvensi SO-100), dan `action.velocity` berisi label kecepatan expert.
Setengah episode memakai noise DART (aksi diganggu, label tetap bersih).
Dataset ini **belum divalidasi** dengan paket `lerobot`, jadi anggap sebagai
"gaya LeRobot".

Instruksi bahasa dibuat dari template ID/EN. Ada dua jenis tugas: taruh di
pad dan tumpuk di kubus.

Demo dari aplikasi bisa dipakai juga. Di tab **Lingkungan**, aktifkan *Rekam*,
lalu *Unduh JSON* atau *Kirim ke bridge*, kemudian jalankan:

```bash
python scripts/import_app_demos.py "data/app_demos/*.json" --out data/app_demos_ds
python -m src.sim.train_pickplace --extra-data data/app_demos_ds
```

## 4. Model baru: policy pick-and-place end-to-end

`src/sim/train_pickplace.py` melatih MLP 17→256×3→5 (≈137 ribu parameter)
dari fitur `src/sim/features.py`. Inputnya: sendi, gripper, posisi ujung, posisi
kubus dan tempat tujuan relatif terhadap ujung, ketinggian kubus, dan jenis
tujuan. Outputnya: kecepatan 4 sendi + perintah gripper. Model ini lalu
dikompresi dengan TTNS χ=16 dan menjalani healing, dengan kontrol healing-only.
Evaluasinya closed-loop di fisika:

* `train_envs`: 4 tempat training, dinamika diacak
* `heldout_env`: pabrik, tempat **dan** servo yang tidak pernah dilihat saat training
* `hard_dynamics`: tempat training dengan parameter **di luar** rentang DR
  (lag 220 ms, latensi 3 langkah, backlash 8 mrad, servo 25% lebih lemah).
  Ini uji sim-to-sim yang mewakili sim-to-real.

Hasil: `results/sim_pickplace.json` dan laporan §6.8
(`docs/technical_report.md`). Model diekspor ke
`app/public/models/pickplace_{ttns,dense}.json` dan dijalankan langsung di
aplikasi (Model → *Pick-place end-to-end*).

Cakupannya: semua dilakukan di simulasi. Posisi objek berasal dari ground truth
yang diberi noise, bukan dari gambar kamera, dan tidak ada robot fisik.

## 5. Mode koneksi aplikasi

| Mode | Perintah |
|---|---|
| Twin fisika MuJoCo via bridge | `python -m src.bridge.server --driver mujoco --env kitchen_counter` → aplikasi: Koneksi → QI-VLA Bridge → `ws://PC:8765/ws`. Scene objek dicerminkan ke aplikasi, sehingga kamera, grounding, dan policy aplikasi bekerja di atas fisika. |
| ROS 2 | `ros2 launch qi_vla_ros sim.launch.py backend:=mujoco` → aplikasi: Koneksi → rosbridge → `ws://PC:9090` (lihat `ros2/qi_vla_ros/README.md`) |
| Robot servo USB / ESP32 | lihat README utama |

## 6. Mobile robot (diff-drive)

Robot: rover seukuran TurtleBot3 Burger (r roda 33 mm, half-track 80 mm,
radius 0,105 m, v ≤ 0,26 m/s, ω ≤ 1,8 rad/s, lidar 360° 36 beam, 3,5 m).
Spesifikasinya ada di `app/public/maps/index.json`.

| Peta | Isi | Status |
|---|---|---|
| `home` (6×5 m) | ruang tamu, dapur, kamar, dock; dua pintu | training |
| `warehouse` (8×6 m) | tiga baris rak, palet, area muat | training |
| `office` (7×5 m) | meja kerja, ruang rapat berpintu, pantry, printer | training |
| `hospital` (8×5 m) | lorong, dua kamar, ruang perawat, apotek; lantai licin, motor lambat | **held-out** |

**Stack navigasi** (`src/sim/mobile/`, dengan padanan JS di `app/src/rover/`),
mengikuti pembagian Nav2:

1. instruksi → tempat (`groundPlace`, alias ID/EN, mis. "pergi ke dapur",
   "go to the charging station")
2. planner global A* di occupancy grid yang di-inflate, dengan biaya clearance
   dan shortcut line-of-sight
3. **costmap**: peta statis ditambah rintangan yang ditandai dari lidar. Ada
   *rintangan tak dikenal* (kotak/orang yang tidak ada di peta), dan robot
   melakukan replanning saat jalur terblokir (dibatasi 1 Hz)
4. pengendali lokal: pure pursuit (expert), atau **policy hasil training**
   (dense / TTNS) dengan input 18 sektor lidar + target pure pursuit + jarak
   tujuan + perintah terakhir
5. **shield lidar** bergaya control barrier function: v ≤ (d_depan − r −
   d_stop)/T, sedangkan putar di tempat selalu diizinkan

**Fisika (MuJoCo)**: dua roda bermotor dengan kontrol kecepatan, dua caster,
inersia, gesekan roda-lantai (slip), lag dan latensi motor, galat radius roda,
noise lidar dan noise lokalisasi. Rangefinder MuJoCo identik dengan raycast 2D
(diuji). Ada juga backend `kinematic` (unicycle) yang cepat.

```bash
python -m src.sim.mobile.train_nav collect --out data/qivla_nav --episodes 600
python -m src.sim.mobile.train_nav collect --out data/qivla_nav_nodr --episodes 600 --no-randomize --seed 100000
python -m src.sim.mobile.train_nav train --seeds 3          # -> results/sim_nav.json, app/public/models/nav_*.json
python -m src.bridge.server --driver rover --map warehouse --unknown 2      # aplikasi: Mobile robot -> Koneksi
ros2 launch qi_vla_ros rover.launch.py backend:=mujoco map:=office          # /cmd_vel, /odom, /scan, TF
```

Set evaluasi: `train_maps`, `heldout_map` (rumah sakit), `hard_dynamics`
(motor 250 ms, latensi 3, lantai licin 0,45, roda −6%), dan
`unknown_obstacles` (3 rintangan tak dikenal di rute). Ada juga ablasi
**tanpa shield**. Metriknya success, tabrakan, SPL, dan % intervensi shield.

**Aplikasi**: pilih *Mobile robot* di header. Mode simulasi memakai model
kinematik + raycast. Mode live tersedia lewat bridge (`--driver rover`, pose
dalam frame peta) atau ROS (`/cmd_vel`, `/odom`, `/scan` ala TurtleBot3; pose
odom disinkronkan ke pose di peta saat terhubung).

## 7. Riset terkait (ringkas, untuk pengembangan berikutnya)

* **LeKiwi** (LeRobot): mobile manipulator murah, lengan SO-101 di atas basis
  omniwheel 3 roda. Ini langkah alami berikutnya: gabungkan rover ini dengan
  lengan di `src/sim` (mobile manipulation). Sumber:
  [Seeed wiki](https://wiki.seeedstudio.com/lerobot_lekiwi/),
  [repo](https://github.com/sigrobotics-uiuc/lekiwi),
  [LeIsaac devices](https://lightwheelai.github.io/leisaac/resources/available_devices).
* **Benchmark VLN** dengan simulasi fisik untuk sim-to-real:
  [VLNVerse](https://arxiv.org/html/2512.19021),
  [SimWorld-Robotics](https://arxiv.org/html/2512.10046v1),
  [POINav](https://arxiv.org/html/2605.28237).
  Ini kandidat evaluasi berbasis kamera, sedangkan stack di sini masih
  berbasis lidar dan peta.
* **Quantum RL untuk navigasi**: hybrid VQC-DDPG dengan lidar
  ([Drăgan dkk., ICAART 2025](https://portal.fis.tum.de/en/publications/continuous-quantum-reinforcement-learning-for-robot-navigation/)),
  quantum DRL humanoid ([arXiv 2509.11388](https://arxiv.org/abs/2509.11388v1)),
  dan [Q-SpiRL](https://arxiv.org/html/2605.20801). Klaim utamanya efisiensi
  parameter di simulasi, belum ada keunggulan di robot nyata. Kandidat
  eksperimen berikutnya: head PQC sebagai pengendali lokal di
  `QiVla/Nav-v0`, dibandingkan dengan MLP berparameter setara.

## 8. Simulator & framework lain

* **MuJoCo viewer**: `python -m mujoco.viewer --mjcf=sim/mjcf/kitchen_counter.xml`
* **Gymnasium** (RL / imitation):
  ```python
  import gymnasium as gym, src.sim.gym_env, src.sim.mobile.gym_env
  env = gym.make("QiVla/PickPlace-v0", env_id="random", randomize=True)
  nav = gym.make("QiVla/Nav-v0", map_id="random", unknown=2)
  ```
* **URDF** (`ros2/qi_vla_ros/urdf/qi_vla_arm.urdf`, `qi_vla_rover.urdf`):
  RViz, Gazebo, Isaac Sim, PyBullet, dan MoveIt/Nav2 (perlu konfigurasi sendiri).
* **LeRobot**: dataset gaya LeRobot memakai konvensi SO-100, sehingga bisa
  dipakai untuk ACT, Diffusion Policy, atau SmolVLA setelah divalidasi dengan
  paket `lerobot`.
