# RealPDE 2026 — İş qeydləri

> Bu fayl çat silinsə belə hər şeyi bərpa etmək üçündür.
> Son yeniləmə: **13 avqust 2026**

İşçi layihə: `D:\Projects\Competitions\realpde\`
Data: `D:\Projects\Competitions\realpde\NeurIPS 2026 RealPDE Competition\train_real\train_real\`
(və `...\train_sim\train_sim\` — tarball-lar iç-içə qovluqla açılır)
Şəkillər: `D:\Projects\Competitions\realpde\figures\`

---

## 0. Dərhal ediləsi (BLOKLAYICI)

| # | İş | Link | Deadline |
|---|---|---|---|
| 1 | Track 1 qeydiyyat formu | https://forms.gle/LYeTgUTfr4ygntNu6 | **20 avqust 2026** |
| 2 | Codabench Track 1 join | https://www.codabench.org/competitions/17363/ | — |
| 3 | (istəyə bağlı) Track 2 formu | https://forms.gle/qmMYaK5u9r86rPGKA | 20 avqust 2026 |
| 4 | Təsdiqdən sonra: Files tab → **starting kit v9** | Codabench | — |

**Codabench-ə join etmək qeydiyyat SAYILMIR.** 20 avqustdan sonra formu doldurmayanların
join sorğusu rədd ediləcək. Starting kit-in olduğu **Files tab yalnız join təsdiqlənəndən
sonra görünür**, ona görə ikisi də lazımdır.

Komanda: **max 3 nəfər**, **bir paylaşılan email/hesab**. Çoxlu hesab = diskvalifikasiya.

### Starting kit nədir və niyə lazımdır

Codabench Files tabından yüklənən rəsmi zip:

| Fayl | Rolu |
|---|---|
| **`scoring.py`** | Leaderboard-un 5 subscore-unu **eynilə** hesablayır — bizim ən vacib alətimiz |
| `submission_template.py` | `predict()` API şablonu; `submission.py` adına dəyişib zip-lənir |
| `load_baseline.py` | CNO / FNO / Transolver checkpoint yükləyicisi |
| `rpde_baselines/` | Həmin modellərin kodu (konteynerdə internet yoxdur → vendor edilib) |
| `_vendor/einops/` | Transolver einops tələb edir, konteynerdə yoxdur |
| `pack_ckpt_fp16.py` | 256 MB limitinə sığmaq üçün (yalnız FNO-ya lazım: 403→201 MB) |
| `smoke_test_kit.py` | Təmiz mühitdə import + forward yoxlaması |
| `README.md` | **Checkpoint hiperparametrləri** — repodakı yaml-lar bunlar üçün etibarsızdır |

Əvəzedilməz iki şey: **`scoring.py`** (MVPE probe nöqtələri və SPS maska məntiqi başqa
heç yerdə sənədləşdirilməyib) və **`mean_std_real.pt`** (rəsmi normalizasiya statistikası).
Bunları təxmin etmək olmaz.

---

## 1. Vaxt vəziyyəti — gecikməmişik

| Tarix | Nə |
|---|---|
| 5 – 19 iyul | Warm-up — bitib, **xalsız idi, heç nə itməyib** |
| **5 avqust** | **Dev Phase RESTART — leaderboard sıfırlandı** |
| **20 avqust** | Qeydiyyat formu deadline |
| 27 sentyabr | Dev Phase bitir (23:59:59 UTC, AoE deyil) |
| 28 sen – 25 okt | Decision Phase: top-10-un metodunu təşkilatçılar **sıfırdan yenidən öyrədir**, private test-də top-3 seçilir |
| 10 noyabr | Nəticələr |
| 25 noyabr | Kod + fact sheet deadline (qaliblər açıq mənbə etməlidir) |
| 6 dekabr | NeurIPS təqdimatı |

**Ən vacib kontekst:** 5 avqustda təşkilatçılar evaluation set-i yenidən qurub fazanı
restart ediblər. Səbəb: window-ları prediction horizon-dan qısa stride ilə kəsmişdilər,
nəticədə bir window-un target-i başqa window-un input-unun içində qalırdı — model ona
artıq verilmiş cavabı tapa bilirdi. Bir neçə komanda bunu bildirib.

Nəticə: **iyul boyu işləyən komandaların leaderboard üstünlüyü silinib.** Biz 8 gün
gecikmişik, 6 həftə yox.

### Limitlər
- **1 submission/gün** dev fazada (warm-up-da 3/gün)
- **100 submission/faza**, **hər iki track arasında paylaşılır**
- Arxiv açıldıqdan sonra **< 256 MB** (checkpoint daxil)
- **5 dəqiqə** wall-clock icra limiti (data download sayılmır)
- `DataLoader` mütləq `num_workers=0`

---

## 2. Hansı PDE?

**Sıxılmayan qeyri-stasionar Navier–Stokes** — NACA4418 aeroprofilinin ətrafındakı
axının 2D en kəsiyi.

- **Re = 3750 – 26700** (sənəddə 2968–27975 yazılıb; faktiki fayllarda yuxarıdakı diapazon)
  → keçid/turbulent rejim, vorteks tökülməsi olan wake. Laminar deyil, xaotik.
- **AoA = 0°, 5°, 10°, 15°, 20°** → 20°-də ayrılma (stall), rejim tamamilə fərqlidir.
- **Real:** su tunelində PIV (Particle Image Velocimetry) ölçmələri.
- **Sim:** eyni həndəsə/şəraitdə 3D CFD.

### Kritik fiziki nüans
Real ölçmə **3D axının bir müstəvi kəsiyidir**. Ona görə müstəvidə divergensiya sıfır
DEYİL (∂u/∂x + ∂v/∂y ≠ 0, çünki müstəviyə perpendikulyar `w` komponenti qalanı daşıyır).

**Modelə sərt "divergence-free" fizika loss-u qoyma** — burada bu yanlışdır və xalı aşağı
salar. Sim data da 3D-nin kəsiyi olduğu üçün eyni vəziyyətdədir.

---

## 3. Data — dəqiq struktur (13 avqust, lokal yoxlanıldı)

Cədvəl deyil, **tensor**. "Sütun" yerinə **kanal** var.

### Codabench formatı (submission bunu görür)
```
input:  (N, T_in=20,  H=32, W=64, C=3)
target: (N, T_out=20, H=32, W=64, C=3)
```
`C = 3` → `[u, v, p]`. Real data-da `p` ölçülməyib, sıfırdır və **xallanmır** —
sıfır qaytarmaq kifayətdir. 32×64 = xam 64×128-in 2× subsample edilmişi.

### `train_real/train_real/{Re}_{AoA}.h5` — 82 fayl, 6.9 GB

| açar | shape | dtype | qeyd |
|---|---|---|---|
| `u` | `(T, 64, 128)` | float64 | axın istiqamətində sürət |
| `v` | `(T, 64, 128)` | float64 | eninə sürət |
| `t` | `(T,)` | float32 | zaman, saniyə |
| `x`, `y` | `(64, 128)` | float64 | grid koordinatları, metr |
| `re` | skalyar | int32 | **faktiki Re — fayl adından fərqlidir** |
| `aoa` | skalyar | int32 | dərəcə |

- 18 fərqli Re × 5 AoA, amma **90-dan 8 kombinasiya yoxdur** (tam grid deyil).
  Çatışmayan: (3750,15), (17775,0), (22875,5), (22875,20), (24150,5), (25425,5), (26700,5), (26700,20)
- `T` = 282 … 868 (orta ≈853). **Qısa trayektoriyalar window kəsməkdə xüsusi baxım istəyir.**
- Xanaların **~10–12%-i həm `u`, həm `v`-də dəqiq sıfırdır** → profil gövdəsi + PIV
  görmə sahəsindən (FOV) kənar. **SPS bunları xallamır.**
- **NaN/Inf yoxdur.**

### `train_sim/train_sim/{Re}_{AoA}.h5` — 100 fayl, 9.2 GB

| açar | shape | dtype |
|---|---|---|
| `u`, `v`, **`p`** | `(1000, 64, 128)` | float32 |
| `t` | `(1000,)` | float64 |
| `x`, `y` | `(64, 128)` | float64 |
| `re`, `aoa` | skalyar | int32 |

### Real ilə sim arasındakı fərqlər (loader bunları həll etməlidir)

| | train_sim | train_real |
|---|---|---|
| Kanallar | `u, v, **p**` | `u, v` |
| dtype | float32 | float64 |
| `T` | sabit 1000 | dəyişkən 282–868 |
| `re` fayl adına uyğun? | **bəli** (10125→10125) | **xeyr** (10125→**10142**) |
| Fayl sayı | 100 | 82 |

**Praktiki tələ:** sim və real-ı işlək nöqtəyə görə cütləşdirmək **fayl adından** getməlidir,
çünki içindəki `re` dəyərləri fərqlidir. Şərtləndirmə (conditioning) üçün isə **içindəki
skalyar** işlədilməlidir. 18 sim faylının real qarşılığı yoxdur.

### ⚠️ İki xəbərdarlıq

1. **`train_real/7575_0.h5` istifadə ETMƏ** — səhvən `6300_0`-ın datasını təkrarlayır
   (12 avqust elanı). Yalnız bu fayl təsirlənib.
2. **HF README yanlışdır:** deyir ki, training tarball-larda `u`/`v` `measured_data/`
   altındadır. **Faktiki olaraq top-level-dədir**, `example_data` ilə eyni. Kodu fərziyyəyə
   görə yazma, açarları yoxla.

### ⚠️ Ən vacib məhdudiyyət
`metadata` **scored çağırışlarda boş dict-dir**. Yəni inference zamanı **Re və AoA
VERİLMİR**. Model rejimi 20 kadrlıq input pəncərəsindən **özü çıxarmalıdır**.
Bu, arxitektura qərarının mərkəzidir.

Həmçinin: `predict` **bir neçə dəfə çağırıla bilər**, hər çağırış ayrı təcrid olunmuş
subprocess-də. Modul səviyyəli state və cache-lər çağırışlar arasında **saxlanmır**.

---

## 4. Xallama — əsas fürsət buradadır

5 subscore var (Rel-L2, TKE, MVPE, Time, SPS), **birləşmə düsturu gizlidir**.
Amma hər birinin öz düsturu açıqdır və bu, çox şey deyir.

### Rel-L2 / TKE / MVPE
```
score = 100 / (1 + 0.5 · error)
```
| error | xal |
|---|---|
| 0.1 | 95 |
| 0.3 | 87 |
| 0.5 | 80 |
| 1.0 | 67 |

**Sıxılmış diapazon.** Modeli 2× yaxşılaşdırsan cəmi 7–8 xal alırsan.

### Time
```
score = 100 / (1 + √r),   r = t_neural / 0.72896s
```
| ms/sample | xal |
|---|---|
| 1 | 96 |
| 5 | 92 |
| 20 | 86 |
| 100 | 73 |

**Kiçik və sürətli model böyük Transolver-i əzir.** 32×64 kimi kiçik gridde bu praktiki
olaraq pulsuz 20 xaldır. Checkpoint yükləmə vaxtı Time-a sayılmır (amma 5 dəq wall-clock
limitinə sayılır).

### SPS — döyüş meydanı
```
pm      = e / (0.5 + e)                    # e = həmin window-un xətası
nil     = (upper − lower) / 0.0563870      # sigma_global, sezon üçün dondurulub
element = (1 − pm) · exp(−nil)  əgər lower ≤ target ≤ upper, əks halda 0
branch  = xallanan elementlərin ortalaması
weighted = 0.5·branch_dm + 0.3·branch_tke + 0.2·branch_mvpe
sps_score = 100 · weighted
```

**Hesablama:** default band (`±5%·|pred|`) ilə tipik SPS ≈ **2–5 xal**. Çünki ±5% intervalı
30% xətası olan proqnozu demək olar heç vaxt örtmür → `inside = False` → element 0 alır.

Öz kalibrlənmiş bound-larını qaytarsan, **eyni modellə SPS ≈ 20–30**.

Optimizasiya sadədir: hər element üçün en `w` seç ki
```
P(örtmə) · exp(−w / 0.0563870)
```
maksimum olsun. Sərbəst axın zonasında dar (xəta kiçik → demək olar 1 xal), wake-də geniş.
**Adaptiv, per-element interval.** Komandaların çoxu bunu görməzdən gələcək, çünki modelin
özü ilə məşğuldurlar.

**Qayda:** bound-lar **hamısı-və-ya-heç biri**. `predict` bir neçə dəfə çağırılır; bəziləri
bound qaytarıb bəziləri qaytarmasa, **hamısı atılır** və default band işlədilir.

### Sıfır xal şərtləri (mütləq qorun)
- Proqnozda NaN/Inf
- Shape `(N, 20, 32, 64, 3)` ilə uyğunsuzluq
- `lower`/`upper` qeyri-sonlu və ya tərsinə (`lower > upper`)

Daha pisi — **run tamamilə uğursuz olur** (xal belə verilmir) əgər bound-ların shape-i
proqnozla uyğun gəlmirsə, və ya proqnozun sample sayı input-la uyğun gəlmirsə.

### Strateji nəticə
**Qalibiyyət yolu ən yaxşı neyron operatorunu qurmaqdan yox,
(a) sürətli kiçik model + (b) düzgün kalibrlənmiş SPS intervalları kombinasiyasından keçir.**

---

## 5. Mühit

`D:\Projects\Competitions\realpde\.venv\` — **evaluation konteyneri ilə eyni versiyalar**:

| | Lokal | Konteyner |
|---|---|---|
| torch | 2.2.2+cu121 ✅ | 2.2.2 + CUDA 12.1 |
| numpy | 1.26.4 ✅ | 1.26 |
| Python | 3.11 | 3.10 |

GPU: **NVIDIA RTX 2060, 6 GB** — CUDA işləyir. 32×64 grid üçün kifayətdir.
CPU: i7-9750H, 32 GB RAM. Disk: D:-də ~150 GB boş.

### Konteynerdə OLMAYAN kitabxanalar
`scipy`, `pandas`, `matplotlib`, **`h5py`**, `einops`, `scikit-learn`, `opencv`

Var: `torch/torchvision/torchaudio 2.2.2`, `numpy 1.26`, `pillow`, `PyYAML`, `requests`,
`tqdm`, `sympy`, `networkx`.

⚠️ **`h5py` yalnız lokal data oxumaq üçündür — `submission.py`-dan ASLA import etmə.**
Saf-Python paketləri arxivə vendor etmək olar; kompilyasiya olunanları yox.

### İşlətmək
```powershell
cd D:\Projects\Competitions\realpde
$py = ".\.venv\Scripts\python.exe"
$D  = ".\NeurIPS 2026 RealPDE Competition"

& $py scripts\inspect_data.py --root "$D"
& $py scripts\verify_stats.py --split-dir "$D\train_real\train_real"
& $py scripts\find_official_split.py
& $py scripts\probe_mask_check.py
& $py scripts\visualize.py --case 10125_10 --frames 100
& $py scripts\visualize.py --case 10125_10 --frames 100 --eval-res --arrow-every 2
& $py scripts\visualize.py --aoa-sweep 10125 --frames 300
& $py scripts\compare_sim_real.py --case 10125_10 --frame 300
```
Qeyd: qovluq 14 avqustda `realpde\` içinə köçürüldü. `visualize.py`,
`probe_mask_check.py`, `compare_sim_real.py` fayllarında `DATA_ROOT` sabit yazılıb;
`inspect_data.py` və `verify_stats.py` isə yolu arqumentlə alır.

---

## 6. `sigma_global` araşdırması — nəticəsiz (dürüst qeyd)

`sigma_global = 0.0563870` sənəddə belə təyin olunub: *"official train_real split-də
u, v kanal standart kənarlaşmalarının ortalaması"*.

**Yenidən çıxara bilmədim.** 82 real trayektoriya üzərində 8 fərqli konvensiya yoxlandı
(xam 64×128 / subsample 32×64) × (maskalı xanalar daxil / xaric) × (7575_0 daxil / xaric):

| variant | nəticə |
|---|---|
| hamısı (8 variant) | 0.0622 – 0.0650 |
| per-trajectory std-lərin ortalaması | 0.0484 |
| **rəsmi hədəf** | **0.0563870** |

Alt çoxluq axtarışında "Re ≤ 21600" ən yaxını çıxdı (0.0556, fərq 0.00075).
**Bu SÜBUT DEYİL** — monoton süpürmə hədəfi mütləq bir yerdə kəsir, yəni nəticə axtarışın
artefaktıdır. Hipotez (gizli validation üçün yüksək-Re halları saxlanılıb) təsdiqlənmədi.

**Bloklamır.** `sigma_global` onsuz da bizə verilib və dondurulub → SPS düsturunda birbaşa
`0.0563870` işlədirik. Əsl normalizasiya statistikası isə starting kit-dəki
**`mean_std_real.pt`**-dədir; onu öz hesabladığımız rəqəmlərlə əvəz etmək olmaz.

Nəticələr: `realpde\data\processed\stats_real.json` (hər faylın mean/std/T-si də orada).

---

## 6b. `scoring.py` oxundu — dəqiq faktlar (13 avqust)

Yer: `realpde_t1_starting_kit_v9\realpde_t1_starting_kit_v9\scoring.py`
(diqqət: qovluq iki dəfə iç-içədir).
**`mean_std_real.pt` T1 kit-də YOXDUR — T2 kit-inin `example_data\` qovluğundadır.**

### SIGMA_GLOBAL-ın mənşəyi (sətir 23–25)
```python
SIGMA_GLOBAL = 0.0563870259   # = (0.09681041 + 0.01596364) / 2
```
| | rəsmi | bizim (82 fayl, maskasız) | fərq |
|---|---|---|---|
| `std_u` | 0.09681041 | 0.108387 | **12%** |
| `std_v` | 0.01596364 | 0.016321 | 2% |

`v` uyğun, `u` uyğun deyil → onların split-i **yüksək-Re hallarını çıxarır**
(Re artdıqca `u` `v`-dən qat-qat çox böyüyür). Sübut deyil, amma güclü işarədir.

### MVPE — probe nöqtələri (sətir 113–143)
Parametrlər: `d=16, center_x=10, center_y=32, n_probe=9, sub_s_real=2`.
Hesablanmış nəticə:
```
probe_x ∈ {13, 21, 29, 37}                  4 stansiya
probe_y ∈ {8,10,12,14,16,18,20,22,24}       9 nöqtə
```
→ **32×64 = 2048 xanadan yalnız 36-sı**, 20 kadr üzrə zaman ortalaması, yalnız `u,v`.
Bir subscore sahənin **~1.8%-indən** gəlir. Training loss-unu bura çəkmək birbaşa xaldır.

### Maskalama — vacib fərq
- `rel_l2`, `tke`, `mvpe`: **maskalanmır**, bütün sahə üzrə (sıfır xanalar daxil)
- **yalnız SPS** `scored = (target != 0.0)` filtri tətbiq edir (element səviyyəsində,
  statik məkan maskası deyil)

### Kanal sayı
`measured_channels()` sıfır olmayan kanalları sayır → real data-da `p=0` olduğu üçün
**`c = 2`**, yalnız `u,v` xallanır.

### Default SPS bandı (sətir 181–183)
```python
interval = 0.1 * abs(p);  lower = p - interval/2;  upper = p + interval/2
```
→ **±5%·|p|**, en = `0.1·|p|`. Əvvəlki hesablamam təsdiqləndi.

### TKE
`kinetic_energy()` `axis=1` (zaman oxu) üzrə işləyir → 20 kadrdan (H,W) xəritəsi çıxır.

### Time
`mean_t_neural_s` **bizdən yox, ingestion proqramından** gəlir — `predictions.npz`-in içində.
`r_min=1.0` → `score = 100/(1+√(t/0.72896))`.

### Səhv idarəetməsi
`_score_submission` istisna atsa, bütün subscore-lar **0** olur və səbəb
`detailed_results.html`-də görünür. Yəni bound shape səhvi = 0 xal, run fail deyil.

---

## 6c. Vizualizasiya və grid həndəsəsi (14 avqust)

Skriptlər: `realpde\scripts\visualize.py`, `realpde\scripts\probe_mask_check.py`
Çıxış: `realpde\figures\`

```powershell
cd D:\Projects\Competitions\realpde
.\.venv\Scripts\python.exe scripts\visualize.py --case 10125_10 --frames 100
.\.venv\Scripts\python.exe scripts\visualize.py --case 10125_10 --frames 100 --eval-res --arrow-every 2
.\.venv\Scripts\python.exe scripts\visualize.py --aoa-sweep 10125 --frames 300
.\.venv\Scripts\python.exe scripts\probe_mask_check.py
```

### Grid həndəsəsi (ölçüldü)
- **dx = dy = 1.711 mm** (tam bircins), sahə **21.9 × 10.8 sm**
- **dt = 0.02 s → 50 Hz**. 20 kadr = **0.4 saniyə** keçmiş, 0.4 saniyə gələcək
- `y` sətir indeksi ilə **azalır** → `dy` mənfidir. Vorticity işarəsi bundan asılıdır
- Profil (maskalı zona) **sol-aşağıda**, wake sağa uzanır
- **Maska zamanla dəyişir** — yəni yalnız profil gövdəsi deyil, PIV dropout-ları da var

### MVPE probe-larının həqiqi yeri
```
probe_x = {13, 21, 29, 37}  →  x = 0.0462, 0.0736, 0.1009, 0.1283 m
probe_y = {8,...,24}        →  y = 0.1078 … 0.1625 m
36 nöqtə = sahənin 1.8%-i
```

**Sənədlərdəki "probe locations behind the airfoil / near the wake region" ifadəsi
DƏQİQ DEYİL.** Şəkildən görünür ki, probe-lar əsasən **profilin üstündə və yaxın
sərbəst-qırxılma (shear) təbəqələrindədir**, wake-də yox. İlk sütun birbaşa profil
gövdəsinin üstünə düşür.

### Probe-ların maskaya düşməsi (82 fayl üzrə ölçüldü)
| probe sütunu | x [m] | maskalı vaxt payı |
|---|---|---|
| 13 | 0.0462 | **24.0%** |
| 21 | 0.0736 | 1.9% |
| 29 | 0.1009 | 0.0% |
| 37 | 0.1283 | 0.0% |

Orta hesabla probe xanalarının **6.5%-i maskalıdır** (bütün sahədə 9.4%).

AoA-ya görə: 0°→8.7%, 5°→10.5%, 10°→5.7%, 15°→5.1%, **20°→3.1%**.
Yəni AoA artdıqca maska AZALIR — gözlənilənin əksi. Fiziki izahı yoxlanmalıdır.

**Nəticə:** MVPE `scoring.py`-da maskalanmır, yəni bu sıfırlar həm surətə, həm məxrəcə
girir. Sıfır olan yerdə sıfır proqnozlaşdırmaq **dəqiq doğrudur** — pulsuz xaldır.
Training loss-unu bu 36 nöqtəyə ağırlaşdırmaq birbaşa MVPE subscore-una təsir edir.

### VSCode qeydi
Redaktorda `h5py` üçün qırmızı xətt görünə bilər — VSCode qlobal Python-a baxır,
bizim venv-ə yox. Kod düzgündür. Düzəltmək üçün interpreter olaraq
`D:\Projects\Competitions\realpde\.venv\Scripts\python.exe` seçilməlidir.

---

## 6d. Sim vs Real — ölçü fərqi (14 avqust, ÖNƏMLİ)

Skript: `realpde\scripts\compare_sim_real.py` → `figures\sim_vs_real_10125_10.png`

Eyni Re, eyni AoA, **eyni grid** (dx, dy, dt, x/y sərhədləri tam eyni) — amma:

| | real | sim |
|---|---|---|
| `mean\|u\|` | 0.126 **m/s** | 0.902 **ölçüsüz** |
| maskalı pay | **9.3%** | **0.3%** |
| görünüş | şumlu, diffuz | təmiz, kəskin |
| `T` | 868 | 1000 |

### Sübut: sim `U∞`-ə bölünüb
AoA=10°-də 18 Re üçün `real/sim` nisbəti ölçüldü:
- **`corr(nisbət, Re) = 0.9954`** — nisbət Re ilə xətti artır
- sim-in `mean|u|` bütün Re-lərdə **~0.86 sabit** qalır
- bundan çıxan vətər: `c = Re·ν/U∞` ≈ **6.3–8.0 sm**, orta **7.2 sm** — sabit, fiziki olaraq tutarlı

→ **sim ölçüsüzdür (u/U∞), real m/s-dədir.**

### İki nəticə
**a)** Sim→real transferində **miqyaslama məcburidir**, yoxsa model 7× fərqlə vuruşur.
**b)** Real `|u| ∝ Re` olduğuna görə model **Re-ni giriş pəncərəsindən özü çıxara bilər**
(sadəcə ortalama sürət böyüklüyündən). `metadata`-nın boş olması problemi bununla həll olunur.

### Maska fərqinin səbəbi
Real-dakı böyük boz zona **profilin özü deyil** — PIV lazerinin kölgəsidir.
Lazer profilə dəyir, arxası işıqlanmır, ölçülmür. Sim-də bu problem yoxdur → orada
yalnız nazik profil gövdəsi maskalıdır.

---

## 6e. Qısa cavablar (sual-cavab)

**S: Bizə hansı kəmiyyətlər verilir?**
`u` = uzununa sürət (axın istiqaməti, x), `v` = eninə sürət (y). Real-da `p` yoxdur
(sıfırdır, xallanmır). Model `u` və `v` proqnozlaşdırır.

**S: Wake nədir?**
Maneənin arxasında qalan pozulmuş axın izi — qayığın arxasındakı zolaq kimi. Axın
soldan sağa gedir, profil solda, wake sağa uzanır. Şəkildəki qırmızı/mavi =
**vorticity** (burulma) işarəsi. Profilin üstündəki nazik mavi və altındakı qırmızı
xətlər **shear (qırxılma) təbəqələridir**.

**S: Yaşıl kvadratlar nədir?**
Fizika deyil. `scoring.py`-ın MVPE metrikasını hesabladığı **36 ölçmə nöqtəsi**.
Kod onları sabit indekslərlə seçir (x = 13,21,29,37), fiziki mənaya görə yox.
Şəkilə ona görə qoyulub ki, harada dayandıqları görünsün → profilin üstündə və
shear təbəqələrində, **wake-də deyil**.

**S: Həm sim, həm real proqnozlaşdırılmalıdır?**
Xeyr. **Yalnız real.** Sim yalnız pretraining üçündür — "Sim2Real" adının mənası budur:
bol və ucuz simulyasiyadan öyrən, az və şumlu real ölçmələrə köçür.

**S: Simulyasiyanın düsturları verilirmi?**
Xeyr. Yalnız çıxış sahələri verilir — solver, mesh, sərhəd şərtləri, turbulentlik
modeli paylaşılmayıb. Alt tənlik sıxılmayan Navier-Stokes-dur, amma **bizə lazım deyil**:
bu, data-driven yarışdır, tənlik həll etmirik. (Bax: bölmə 3-dəki PINN qeydi.)

**S: 20 kadr nə qədər vaxtdır?**
`dt = 0.02 s` → **20 kadr = 0.4 saniyə**. Tapşırıq: 0.4 s keçmişə bax, 0.4 s gələcəyi ver.

---

## 6f. Baseline xalları — STRATEJİ CƏHƏTDƏN ƏN VACİB BÖLMƏ (14 avqust)

`scripts\eval_baselines.py`, bizim val split-imizdə (3 saxlanmış Re, 273 pəncərə),
rəsmi `scoring.py` ilə. `t_neural = 5 ms` fərz edilib.

| baseline | Rel-L2 | TKE | MVPE | Time | SPS |
|---|---|---|---|---|---|
| persistence (son kadrı təkrarla) | 93.50 | **66.67** | 93.39 | 92.35 | 17.01 |
| last_window (giriş pəncərəsini təkrarla) | 92.70 | 72.98 | 94.40 | 92.35 | 16.03 |
| time_mean (pəncərənin zaman ortası) | **94.03** | **66.67** | 94.40 | 92.35 | 17.50 |
| zero (hamısı sıfır) | 66.67 | 66.67 | 66.67 | 92.35 | 0.00 |
| **oracle (pred = target)** | 100 | 100 | 100 | 92.35 | **87.70** |

### Nəticə: mövcud fərq (headroom)
| Metrika | trivial | oracle | **fərq** |
|---|---|---|---|
| Rel-L2 | 94.0 | 100 | **6** |
| MVPE | 94.4 | 100 | **6** |
| Time | 92.4 (5ms) | 96 (1ms) | **4** |
| **TKE** | **66.7** | 100 | **33** |
| **SPS** | **17.5** | 87.7 (default band) → ~100 (öz bound) | **~70** |

### Üç kritik müşahidə

**1. Rel-L2 və MVPE DOYMUŞDUR.** Giriş pəncərəsinin zaman ortasını qaytarmaq
Rel-L2 = 94.03 verir. Səbəb: nisbi L2-nin məxrəci böyük **orta axınla** doludur,
o isə 0.4 saniyədə demək olar dəyişmir. Bu metrikaları qovmaq vaxt itkisidir.

**2. TKE zamanla sabit proqnozu SIFIRA vurur.** persistence və time_mean —
hər ikisi TKE xətası **dəqiq 1.0**, yəni **sıfır qaytarmaqla eyni xal**.
TKE məhz sürət *dalğalanmalarının* enerjisidir; hamar proqnozda dalğalanma yoxdur.

**3. Oracle-ın SPS-i 100 deyil, 87.7-dir.** Mükəmməl proqnozda coverage = 1.0,
yəni itki tamamilə **band eninin cəzasından** (`exp(-nil)`) gəlir. Öz dar
bound-larımızla bu tavan ~100-ə qalxır.

### Strateji nəticə
Komandaların əksəriyyəti MSE minimallaşdıracaq. MSE **şərti ortanı** öyrədir →
hamar, dalğalanmasız proqnoz → Rel-L2 yaxşılaşır (onsuz da doymuş) və
**TKE ölür**. Klassik video-proqnoz problemi.

### TKE-nin dəqiqləşdirilməsi (vizual sübut)
`scripts\visualize_prediction.py` → `figures\pred_*_tke.png`

| baseline | saxlanan TKE | TKE xalı |
|---|---|---|
| `time_mean` | **0.0%** | 66.7 |
| `last_window` | **121.9%** | 65.9 |

`last_window` TKE-nin miqdarını (hətta artığını) saxlayır, amma xal yenə aşağıdır.
Səbəb: TKE metrikası TKE **sahəsinin** nisbi L2-sidir → dalğalanma **düzgün yerdə**
olmalıdır. "Modelə şum əlavə edək" yanaşması İŞLƏMƏYƏCƏK.

→ **Bizim üstünlüyümüz TKE + SPS-dədir, Rel-L2-də yox.**
→ Loss funksiyası TKE-ni birbaşa hədəfləməlidir (yalnız MSE yox).
→ Kalibrlənmiş `lower`/`upper` bound-lar ~70 xallıq açıq sahədir.

⚠️ Qeyd: bunlar bizim val split-imizdədir, gizli set fərqli ola bilər.
Amma metrikaların **quruluşu** universaldır.

---

## 6g. Data pipeline (qurulub, 14 avqust)

```
scripts\prepare_data.py    .h5 → 32×64 float32 cache (real 1.05 GB, sim 2.29 GB)
src\realpde\data.py        pəncərələmə, split, normalizasiya
src\realpde\local_score.py rəsmi scoring.py-ı İDXAL edir (təkrar yazmır)
scripts\test_data.py       sağlamlıq yoxlamaları + sızma testi
scripts\eval_baselines.py  yuxarıdakı cədvəl
```

- **real**: 81 trayektoriya (7575_0 xaric), 69 085 kadr, kanallar `[u,v]`
- **sim**: 100 trayektoriya, 100 000 kadr, kanallar `[u,v,p]`
- Split: **Re üzrə** — val Re = {6300, 13950, 22875}, train 15 Re / 68 case,
  val 3 Re / 13 case. Gizli test "görünməmiş Re" işlətdiyi üçün val da elə qurulub.
- **val_stride = 40** → pəncərələr tam ayrıdır, `test_data.py` sızmanın **0** olduğunu
  təsdiqləyir. train_stride = 10 (orada üst-üstə düşmə zərərsizdir, augmentasiyadır).

### Sim→real miqyas körpüsü
`U∞ = Re × 1.394e-5` (ölçülmüş sabit) ilə sim m/s-ə çevrilir, sonra hər ikisi
rəsmi statistika ilə normallaşdırılır.
- **əvvəl**: normallaşdırılmış sim `u` ortalaması **+7.3** (real: +0.02)
- **sonra**: **+0.37** → boşluq **20× azaldı**
Qalan fərq əsl sim2real gap-dır; onu model bağlamalıdır, preprocessing yox.

### Rəsmi normalizasiya (T2 kit `example_data\mean_std_real.pt`)
```
mean = [0.1550, -0.0005, 0.0]
std  = [0.0968,  0.0160, 0.0]   ← (0.0968+0.0160)/2 = SIGMA_GLOBAL ✓
```
`p` kanalında std = 0 olduğu üçün kodda 1.0-a əvəz olunur (sıfıra bölməmək üçün).

---

## 6h. Sızma auditi (14 avqust) — TƏMİZ

`scripts\leakage_audit.py`

| Yoxlama | Nəticə |
|---|---|
| train/val ortaq case | **yoxdur** |
| train/val ortaq Re | **yoxdur** |
| train/val ortaq trayektoriya offset | **yoxdur** (fərqli fayllar) |
| val-da window-lar arası sızma | **0** |
| release-də ikinci gizli dublikat fayl | **tapılmadı** |

### Rel-L2-nin niyə "yumşaq" olduğu — rəqəmlə
```
||target||        = 32.08
||zaman ortası||  = 31.95   →  enerjinin 99.2%-i
||dalğalanma||    =  2.66   →  enerjinin  0.8%-i
```
Hədəfin enerjisinin **99.2%-i sabit orta axındır**, o isə 0.4 s-də tərpənmir.
Ona görə girişi kopyalamaq 90-larda xal alır. **Bu, sızma deyil, metrikanın quruluşudur.**

Zaman ortasını mükəmməl tapıb dalğalanmanı sıfır versən: `rel_l2 = 95.91`, `tke = 66.67`.
→ Dalğalanmanı modelləşdirməkdən **Rel-L2-də ~4 xal**, **TKE-də 33 xal** qazanılır.

Maskalı xanalar (9.6%) nisbətin həm surətinə, həm məxrəcinə sıfır verir → xalı nə
şişirdir, nə azaldır; sadəcə effektiv sahəni kiçildir.

---

## 6i. LÜĞƏT — metrikalar və terminlər

### Beş metrika

**`rel_l2_score` — Relative L2 (nisbi L2 xətası)**
`||proqnoz − həqiqət|| / ||həqiqət||`, sonra `100/(1+0.5·xəta)`.
Ən adi "piksel-piksel fərq" ölçüsü. Burada **yumşaqdır**, çünki məxrəc böyük sabit
orta axınla doludur (yuxarı bax). Trivial baseline 94 alır.

**`tke_score` — Turbulent Kinetic Energy (turbulent kinetik enerji)**
`TKE = 0.5·(zaman_dispersiyası(u) + zaman_dispersiyası(v))`.
Sürətin **orta qiymətdən sapmalarının** enerjisi — yəni axının nə qədər "çalxalandığı".
Sonra bu TKE **xəritəsinin** nisbi L2 xətası götürülür.
→ Zamanla sabit proqnozda dispersiya = 0 → TKE = 0 → xəta = 1.0 → xal 66.67 (döşəmə).
→ Miqdarı düz tapmaq bəs etmir, **məkanı da** düz olmalıdır.

**`mvpe_score` — Mean Velocity Profile Error (orta sürət profili xətası)**
Yalnız **36 nöqtədə** (x = 13,21,29,37; y = 8..24), 20 kadr üzrə zaman ortalaması
alınmış `u,v` müqayisə olunur. Sahənin 1.8%-i. Trivial baseline 94 alır.

**`time_score` — sürət**
`100/(1+√(t/0.72896))`, `t` = bir nümunəyə düşən inference saniyəsi.
`0.72896 s` = istinad ədədi həlledicinin vaxtı. Checkpoint yükləmə sayılmır.

**`sps_score` — Safe Prediction Score (etibarlı proqnoz xalı)**
Yalnız proqnozu yox, **proqnozun etibarlılıq intervalını** qiymətləndirir.
Hər element üçün `lower`/`upper` verirsən; xal iki şeydən asılıdır:
- **coverage** — həqiqət intervalın içindədirmi (`inside`)
- **tightness** — interval nə qədər dardır (`nil = (upper−lower)/0.0563870`)
`element = (1−pm)·exp(−nil)`, əgər içindədirsə; əks halda 0.
Geniş interval → örtür, amma `exp(−nil)` kiçilir. Dar interval → örtmür, 0 alır.
→ **Optimal en tapılmalıdır.** Verməsən default `±5%·|proqnoz|` işlədilir və ~17 xal alırsan.

**`final_score`** — bu beşinin birləşməsi. **Düsturu açıqlanmayıb.** Leaderboard yalnız
bunu göstərir; öz submission-unun detallarında beş subscore görünür.

### Fizika terminləri

| Termin | İzah |
|---|---|
| **PDE** | Xüsusi törəməli diferensial tənlik. Burada: Navier–Stokes |
| **Navier–Stokes** | Mayenin hərəkətini təsvir edən əsas tənlik |
| **sıxılmayan** | Sıxlıq sabitdir (su üçün doğru) |
| **PIV** | Particle Image Velocimetry — suya zərrəciklər səpilir, lazerlə işıqlandırılır, kameralarla izlənir. Sürət sahəsi belə **ölçülür** |
| **CFD** | Computational Fluid Dynamics — tənliyin kompüterdə **həll edilməsi** (simulyasiya) |
| **NACA4418** | Standart aeroprofil forması (qanad kəsiyi) |
| **Re (Reynolds ədədi)** | `Re = U·c/ν`. Ətalət/özlülük nisbəti. Kiçik Re = hamar (laminar), böyük Re = çalxantılı (turbulent) |
| **AoA (hücum bucağı)** | Profilin axına nisbətən bucağı |
| **vətər (chord)** | Profilin uzunluğu. Burada ≈ **7.2 sm** (ölçdük) |
| **U∞ (freestream)** | Maneədən uzaqdakı sərbəst axın sürəti |
| **wake (iz)** | Maneənin **arxasında** qalan pozulmuş, yavaşımış, burulğanlı zolaq. Qayığın arxasındakı iz kimi |
| **vorticity (burulma)** | `ω = ∂v/∂x − ∂u/∂y`. Mayenin yerli fırlanma sürəti. Şəkillərdə qırmızı/mavi |
| **shear layer (qırxılma təbəqəsi)** | Sürətin kəskin dəyişdiyi nazik zolaq — profilin səthində |
| **stall (ayrılma)** | Böyük AoA-da axının səthdən qopması. Rejim tamamilə dəyişir |
| **u, v** | Sürətin uzununa (x) və eninə (y) komponentləri |
| **p** | Təzyiq. Real data-da **ölçülməyib** |

### ML terminləri

| Termin | İzah |
|---|---|
| **neural operator** | Funksiyadan funksiyaya təsvir öyrənən şəbəkə. Burada: sahə → sahə |
| **FNO** | Fourier Neural Operator. Fourier fəzasında işləyir |
| **CNO** | Convolutional Neural Operator |
| **Transolver** | Transformer əsaslı operator |
| **autoregressive** | Bir addım proqnozlaşdır, onu geri ver, təkrarla. Xəta yığılır |
| **direct (birbaşa)** | 20 kadrı bir dəfəyə ver. Sürətli, xəta yığılmır. **Bizim seçimimiz** |
| **pretraining** | Bol data-da (sim) ilkin öyrətmə |
| **finetuning** | Az və hədəf data-da (real) dəqiqləşdirmə |
| **Sim2Real** | Simulyasiyada öyrənib real ölçmələrə köçürmək. **Track 1-in mahiyyəti** |
| **window / stride** | Pəncərə = 20 giriş + 20 çıxış kadr. Stride = pəncərələr arası addım |
| **data leakage (sızma)** | Modelin cavabı "görməməli olduğu yerdən" alması. Təşkilatçılar buna düşüb |
| **overfitting** | Modelin training data-nı əzbərləyib yeni data-da uğursuz olması |
| **baseline** | Müqayisə həddi. Trivial baseline = "heç nə etməsək nə alarıq" |
| **oracle** | Mükəmməl proqnoz. Tavanı göstərir |
| **coverage (örtmə)** | Həqiqətin verilən intervalın içində olma payı |
| **calibration (kalibrləmə)** | İntervalın "dürüst" olması: 90% interval həqiqətən 90% hallarda örtməlidir |
| **normalization** | Data-nı `(x−mean)/std` ilə eyni miqyasa gətirmək |
| **checkpoint** | Öyrədilmiş model çəkiləri faylı |
| **MSE** | Mean Squared Error. **Şərti ortanı** öyrədir → hamar proqnoz → TKE ölür |

### Yarış terminləri

| Termin | İzah |
|---|---|
| **Codabench** | Submission göndərilən platforma |
| **starting kit** | Rəsmi başlanğıc paketi (bölmə 0-a bax) |
| **leaderboard** | Sıralama lövhəsi. Yalnız `final_score` göstərir |
| **Development Phase** | Əsas iş fazası. 27 sentyabrda bitir |
| **Decision Phase** | Top-10-un metodunun təşkilatçılar tərəfindən **sıfırdan yenidən öyrədilməsi** |
| **private test set** | Gizli test dəsti — **görünməmiş Re/AoA** ilə |
| **ingestion program** | Codabench-də bizim `predict()`-i çağıran və vaxtı ölçən proqram |

---

## 6j. Şəkillər — hansı fayl nəyi göstərir

### "Vizualizasiya" burada nə deməkdir
Bir data faylında (`10125_10.h5`) **868 an**, hər an üçün **64×128 tor**, hər xanada
**2 rəqəm** (`u`, `v`) var → **14 milyon rəqəm**. Cədvələ baxıb heç nə anlamaq olmur.
Ona görə rəqəmlər **rəngə çevrilir**: hər an üçün bir şəkil, şəkillər ardıcıl
düzülüb **GIF video** olur. Model yoxdur, proqnoz yoxdur — mövcud data görünən halda.

Ad açılışı: `10125_10` = **Re 10125, AoA 10°**. `w100` = validation-un 100-cü pəncərəsi.

### A qrupu — sadəcə DATA (proqnoz yoxdur)
| Fayl | Nə göstərir |
|---|---|
| `real_10125_10.gif` | Real PIV ölçməsinin videosu, xam 64×128 |
| `real_10125_10_eval32x64.gif` | Eyni video 32×64-də — **modelin əslində görəcəyi** |
| `real_10125_aoa_sweep.png` | 5 sabit şəkil: eyni Re, beş AoA (0..20°) |
| `sim_vs_real_10125_10.png` | Simulyasiya ilə real ölçmə yan-yana |

### B qrupu — baseline testi (hələ də MODEL YOXDUR)
Sual: *"heç nə öyrənmədən, ibtidai arifmetika ilə nə qədər xal alarıq?"*
- `time_mean` = giriş 20 kadrın **ortası**, 20 dəfə təkrarlanmış
- `last_window` = giriş 20 kadr **olduğu kimi kopyalanmış**

| Fayl | Nə göstərir |
|---|---|
| `pred_*_w100.gif` | 3 zolaq: **həqiqət / ibtidai üsulun cavabı / fərq** |
| `pred_*_w100_tke.png` | TKE xəritəsi: yuxarıda həqiqət, aşağıda ibtidai üsul |

**Niyə vacibdir:** `time_mean`-in TKE şəklində yuxarı zolaq parlaq, aşağı zolaq
**tamamilə qaradır**. Yəni kadrların ortasını alsan, axının çalxalanması itir — və
`tke_score` məhz onu ölçür. Bu, loss funksiyasını adi MSE-dən fərqli qurmaq
qərarının səbəbidir.

---

## 6k. Re və AoA — harada, necə

**Hər fayl = bir sabit təcrübə.** `10125_10.h5`-də 868 anın **hamısında**
Re = 10142, AoA = 10°. Trayektoriya boyu **dəyişmir** (`shape ()` = tək ədəd).
"AoA dəyişir" ifadəsi **fayllar arasını** bildirir: 18 Re × 5 AoA = 82 fayl.

**İki yerdə qeyd olunub:**
1. Fayl adında → `{Re}_{AoA}.h5`
2. Faylın içində → `re`, `aoa` skalyar açarları

⚠️ **Üst-üstə düşmürlər:** fayl adı `10125`, içindəki `re` = `10142`.
Fayl adı **nominal** (planlaşdırılmış), içindəki **faktiki ölçülmüş** dəyərdir.
Simulyasiyada ikisi eynidir.
→ Sim və real-ı cütləşdirmək **fayl adı** ilə, şərtləndirmə **içindəki dəyər** ilə.

⚠️ **Submission zamanı nə Re, nə AoA verilir** (`metadata` boş sözlük).
Model rejimi girişdən özü çıxarmalıdır. Yolu var: real data-da `|u| ∝ Re` (bölmə 6d).

### Digər sabitlər
| Kəmiyyət | Dəyər | Mənbə |
|---|---|---|
| `dx`, `dy` | 1.711 mm | `x`, `y` massivlərindən ölçüldü |
| `dt` | 0.02 s (50 Hz) | `t` massivindən |
| sahə | 21.9 × 10.8 sm | `x`, `y` diapazonu |
| vətər (chord) | ≈ 7.2 sm | `Re·ν/U∞`-dən çıxarıldı |
| `ν` (su) | ≈ 1e-6 m²/s | fiziki sabit |
| `U∞` | `Re × 1.394e-5` m/s | ölçüldü (bölmə 6d) |

Sərhəd şərtləri, mesh, turbulentlik modeli — **verilmir**.

---

## 6l. Bu tip məsələlərdə hansı modellər işlədilir

RealPDEBench-in öz 10 baseline-ı: **FNO, CNO, Transolver, U-Net, WDNO, DeepONet,
MWT, GK-Transformer, DPOT, DMD**. Yarışın kit-i bunlardan üçünü verir (FNO, CNO, Transolver).

| Ailə | Nümunələr | Xüsusiyyət | Bizim üçün |
|---|---|---|---|
| **Neural operator** | FNO, DeepONet, MWT, GNO | Funksiya→funksiya təsviri, ayırdetmədən asılı deyil. FNO Fourier fəzasında işləyir, qlobal görmə sahəsi | **Uyğun** — FNO kit-də var |
| **Konvolyusion** | U-Net, ResNet | Sadə, sürətli, müntəzəm torda güclü | **Ən uyğun** — torumuz 32×64 müntəzəmdir |
| **Transformer** | Transolver, GK-Transformer, OFormer, DPOT | Güclü, amma **yavaş** | Time xalını aşağı salır — ehtiyatlı |
| **Generativ** | diffusion, WDNO | **Hamarlıq problemini həll edir** — MSE-nin bulanıq proqnozuna qarşı standart cavab | **TKE üçün maraqlı** |
| **Klassik** | DMD, POD-Galerkin | Ucuz, şərh edilə bilən, **kvazi-periodik** axında güclü | Vorteks tökülməsi periodikdir → baxmağa dəyər |

### Bizim məhdudiyyətlərə görə seçim
- Tor **kiçik və müntəzəmdir** (32×64) → U-Net / FNO ideal
- `time_score` **sürətə** görə verilir → transformer-dən qaçırıq
- **256 MB** limiti → kiçik model
- **TKE dalğalanma tələb edir** → təkcə MSE kifayət deyil

→ **Başlanğıc: kiçik U-Net və ya FNO, birbaşa 40→40 kanal, MSE + TKE həddi.**
Vaxt qalsa generativ təkmilləşdirmə.

---

## 6m. İLK SUBMISSION — leaderboard nəticəsi (14 avqust)

Fayl: `submissions\unet_v1.zip` (kiçik U-Net, yalnız MSE, 1.76M parametr)

**Nəticə: `final_score = 73.664598`, 138 komanda içində 102-ci yer.**

| subscore | dəyər |
|---|---|
| rel_l2_score | 94.634455 |
| tke_score | 73.676616 |
| mvpe_score | 93.159743 |
| time_score | 92.302811 |
| **sps_score** | **16.644090** |

### `final_score` düsturu — hələ naməlum
| namizəd | dəyər | nəticə |
|---|---|---|
| 5-in arifmetik ortası | 74.0835 | ✗ (fərq −0.419) |
| həndəsi orta | 63.07 | ✗ |
| harmonik orta | 47.26 | ✗ |
| minimum | 16.64 | ✗ |

Arifmetik ortaya **yaxındır**, sapma aşağı xallara (sps, tke) bir az artıq çəki verildiyini
göstərir. **Bir submission = bir tənlik, beş naməlum** → çəkiləri təyin etmək mümkün deyil.
Gündə 1 submission limiti ilə bunu qəsdən araşdırmaq sərfəli deyil.

### ⭐ Lokal ↔ gizli set kalibrləməsi (ÇOX VACİB)
| metrika | bizim val | gizli set | fərq |
|---|---|---|---|
| rel_l2 | 96.06 | 94.63 | **−1.43** |
| tke | 75.37 | 73.68 | **−1.69** |
| mvpe | 96.10 | 93.16 | **−2.94** |
| sps | 22.07 | 16.64 | **−5.43** |

**Lokal rəqəmlərimiz nikbindir.** Bundan sonra lokal nəticəyə bu düzəlişlə baxmaq lazımdır.
SPS-dəki 5.4 xallıq fərq ən böyüyüdür — gizli setdə maskalama/pəncərə seçimi fərqli ola bilər.

### Sürət
`time_score = 92.30` → platformada **5.07 ms/nümunə**. Bizim lokal GPU ölçümümüz **1.2 ms**.
Fərq çağırış başına əlavə yükdəndir. Orada da ~4 xal var, amma prioritet deyil.

### Qalan boşluqlar
| metrika | indiki | boşluq |
|---|---|---|
| rel_l2 | 94.63 | 5.4 |
| mvpe | 93.16 | 6.8 |
| time | 92.30 | 7.7 |
| tke | 73.68 | 26.3 |
| **sps** | **16.64** | **~83** |

**SPS tək başına hamısından üstündür.** Çəki 0.15 olsa belə 16.6→60 keçid final-a **+6.5**
verir; digər dördünü tavana çatdırmaq isə cəmi **+4**. Üstəlik SPS training tələb etmir.

---

## 6n. Sınanmış və UĞURSUZ olmuş ideyalar

Bunları təkrar sınamamaq üçün yazıram.

### 1. Loss-a TKE həddi əlavə etmək — UĞURSUZ
| `w_tke` | rel_l2 | tke | seçim |
|---|---|---|---|
| 0.0 | 96.06 | **75.37** | **89.17** |
| 0.5 | 95.45 | 74.87 | 88.67 |
| 2.0 | 93.91 | 73.90 | 87.27 |

Çəki artdıqca `tke_score` **monoton düşür** — özü də daxil olmaqla hər şey pisləşir.
`w_tke=2.0` 5-ci epoxadan sonra pisləşməyə başlayır (qeyri-sabit).
**Səbəb:** TKE ikinci tərtib statistikadır; gradienti "harada nə qədər" deyir, amma
"**hansı anda**" demir. MSE-nin öyrənməsini pozur, əvəzinə heç nə vermir.

### 2. Semi-Laqranj daşınma prioru — UĞURSUZ
| üsul | rel_l2 | tke_err |
|---|---|---|
| persistence | **93.50** | **1.00** |
| özünü-daşıma (substeps=4) | 90.06 | 2.42 |
| donmuş: pəncərə orta V | 86.13 | 8.50 |
| donmuş: pəncərə orta V + maska | 91.70 | 2.94 |

**Bütün variantlar persistence-dən pisdir.** İki səbəb:
1. **Nisbi L2-nin 99.2%-i orta axındır.** Persistence onu dəqiq saxlayır; daşınma
   bilinear interpolyasiya ilə korlayır. 0.8%-i düzəltmək üçün 99.2%-i riskə atmaq.
2. **Daşınma məkan strukturunu zaman dispersiyasına çevirir.** Sabit sürətlə sürüşdürəndə
   nöqtənin yanından məkan naxışı keçir → süni TKE → `tke_err` 8.5-ə qalxır.

Fizika arqumenti (yüksək Re-də 34 xana daşınma) **doğrudur**, amma sərt prior kimi tətbiq
etmək yanlışdır: şəbəkə daşınmanı onsuz da öyrənir, özü də interpolyasiya itkisi olmadan.

---

## 6o. Model hesabatı — hər model üçün bir şəkil

`scripts\model_report.py --checkpoint <ckpt>` → `figures\report_<ad>.png`

Altı panel: burulma (həqiqət/proqnoz/fərq, 20-ci kadr) · TKE xəritələri · **məkan xəta
xəritəsi** · xəta-lead time · korrelyasiya/amplituda-lead time · xal cədvəli.

### İki tapıntı
**1. Xətanın demək olar hamısı wake zolağındadır.** Məkan xəta xəritəsi bunu birmənalı
göstərir: sərbəst axın tünddür, profilin arxasındakı zolaq parlaqdır.

**2. `amplituda ≈ korrelyasiya`** (hər ikisi 0.87 → 0.55). Bu təsadüf deyil:
**MSE-optimal proqnozverici dalğalanmanı məhz korrelyasiya əmsalı qədər kiçildir.**
Yəni model səhv etmir, MSE-nin tələb etdiyini edir.
→ `α = 1/corr ≈ 1.54` ilə amplitudanı bərpa etmək **nəzəri olaraq əsaslıdır**.

---

## 6p. Advective model, pretraining, model ölçüsü (14-15 avqust)

`scripts\compare_runs.py` — bütün run-ları yan-yana qoyur.

| run | arxitektura | base | pretrain | rel_l2 | tke | mvpe | sps | time | **seçim** |
|---|---|---|---|---|---|---|---|---|---|
| persistence | — | — | — | 93.50 | 66.67 | 93.39 | 17.01 | 92.35 | 84.520 |
| `tke0` (1-ci submission) | unet | 64 | — | 96.06 | 75.37 | 96.10 | 22.07 | 95.84 | 89.174 |
| `advective` | advective | 64 | — | 96.16 | 75.78 | 96.42 | 22.41 | 88.26 | 89.456 |
| `adv_base128` | advective | **128** | — | 96.23 | 76.05 | 96.46 | 22.82 | 88.08 | 89.580 |
| **`adv_finetuned`** | advective | 64 | **sim** | **96.25** | **76.32** | **96.54** | 22.96 | 91.41 | **89.703** |

### Nəticələr
**1. Advection girişi işləyir** (istifadəçinin ideyası): `unet` 89.174 → `advective` 89.456.
Model `advect_sequence` işlədir — sahə **öz sürəti ilə** daşınır, sürət hər addımda
yenidən oxunur, ona görə **burulğanların fırlanması ötürülür**. Girişə həmçinin
`|V|`, burulma, divergensiya və mütləq `x,y` koordinatları verilir.

**2. Sim pretraining kömək edir**, amma az: **+0.248**. Track 1-in mahiyyəti budur.
Diqqətçəkən: `amplituda (0.731) > korrelyasiya (0.626)` — əvvəlki modeldə ikisi
**bərabər** idi (MSE-nin nəzəri davranışı). Pretraining modelin dalğalanma
amplitudasını qaldırıb, saxlanan TKE 44.1% → **55.2%**.

**3. Model böyütmək İŞLƏMİR**: base 64→128 (1.79M→7.04M) cəmi **+0.125** verir,
əvəzinə 4× checkpoint və ən pis `time_score`. **Darboğaz tutum deyil, data azlığıdır** —
ona görə sim data (data əlavə edir) kömək etdi, tutum artırmaq isə yox.

⚠️ Hər iki qazanc (+0.248, +0.125) val split-imizin **daxili yayılmasından (3-4 xal)
kiçikdir** — yəni ciddi mənada hələ sübut olunmuş deyil.

---

## 6q. ⭐ SPS KALİBRLƏMƏSİ — ƏN BÖYÜK QAZANC (15 avqust)

`scripts\calibrate_sps.py`, `src\realpde\sps.py`

### Problem
1-ci submission `lower`/`upper` **vermirdi**, ona görə scorer öz default band-ını
işlədirdi: `±5%·|proqnoz|`. O, modelin xətası ilə **heç bir əlaqəsi olmayan** ixtiyari
endir. Ölçüldü: elementlərin **60.1%-i intervaldan kənarda qalır və dəqiq sıfır alır.**

Səbəb: band `|proqnoz|`-un faizidir → sürətin kiçik olduğu yerdə dar olur.
Sürətin kiçik olduğu yer isə məhz **wake**-dir, yəni xətanın ən böyük olduğu yer.

### Modelin əsl xətası
```
|qalıq| ortalama 0.00510   median 0.00219   p90 0.01190
qalıq std        0.01074
SIGMA_GLOBAL     0.05639   ← scorer-in miqyası
```
Xətamız `SIGMA_GLOBAL`-dan **5 dəfə kiçikdir** → dar VƏ örtən interval mümkündür.

### Riyazi optimum
Qalıq `N(0, s²)` olsa, `exp(-w/σ)·P(|e| ≤ w/2)` maksimumu:
```
φ(r) / (2Φ(r) − 1) = s / SIGMA_GLOBAL,    optimal en w = 2·s·r
```
Yəni optimal en **yalnız `k = s/σ` nisbətindən** asılıdır. Kod: `src\realpde\sps.py`.
`scipy` konteynerdə yoxdur → analitik erf yaxınlaşması (Abramowitz–Stegun) + cədvəl.

⚠️ **Performans tələsi:** ilk versiyada `np.vectorize(math.erf)` işlətmişdim — o,
Python döngüsüdür və 22M element üzərində **heç vaxt bitmir**. İndi 2.2M element 0.28 s.

### Nəticələr (273 validation pəncərəsi)
| üsul | SPS | örtmə | orta en |
|---|---|---|---|
| default band (1-ci submission) | **22.96** | 0.399 | 0.00822 |
| training-dən öyrənilmiş, ×0.8 | **48.74** | 0.876 | 0.01784 |
| training-dən, ×1.0 | 47.39 | 0.915 | 0.02230 |
| enerjiyə görə miqyaslanmış ×0.8 | 48.54 | 0.841 | 0.01579 |
| *validation-dan öyrənilmiş ×0.8 (aldatma)* | *48.97* | *0.865* | *0.01687* |

**Dürüst versiya aldadıcıdan cəmi 0.23 xal geridədir** → `sigma` xəritəsi görünməmiş
Reynolds ədədlərinə yaxşı ümumiləşir. Gizli test üçün ən vacib sual bu idi.

Enerjiyə görə miqyaslama **kömək etmədi** → sadə sabit xəritə seçildi.
`sigma` faylı: `checkpoints\sps_sigma.npz` (289 KB), submission-a qoşulur.

**Qazanc: +25.78 xal, training tələb etmədən.**

---

## 6r. İKİNCİ SUBMISSION — hazır və yoxlanılıb

```
submissions\adv_v2.zip   6.64 MB   (model: adv_finetuned + kalibrlənmiş bound-lar)
```

| subscore | `unet_v1` (göndərildi) | `adv_v2` (hazır) | fərq |
|---|---|---|---|
| rel_l2 | 96.06 | 96.25 | +0.19 |
| tke | 75.37 | 76.32 | +0.95 |
| mvpe | 96.10 | 96.54 | +0.44 |
| **sps** | 22.07 | **48.74** | **+26.67** |
| time | 95.84 | 86.54 | −9.30 |

`time` düşür, çünki `advect_sequence` ardıcıldır (40 `grid_sample` çağırışı).
Bu, bilərəkdən qəbul edilmiş mübadilədir: dəqiqlik prioritetdir.

### ⚠️ Paketləmə səhvi tutuldu
`build_submission.py`-ın təmiz-proses yoxlaması `submission.py`-ın hələ də
`UNetForecaster` yüklədiyini aşkarladı, checkpoint isə `AdvectiveUNet` idi →
shape uyğunsuzluğu. **Göndərilsəydi bütün subscore-lar sıfır olardı və bir günlük
haqqımız yanardı.** Düzəliş: arxitektura adı checkpoint-in `config`-inə yazılır.

Bu, yoxlama harness-inin niyə qurulduğunun konkret sübutudur.

---

## 6s. İnfrastruktur qeydləri (15 avqust)

**`evaluate()` kanal səhvi:** training döngüsü `[..., :use_channels]` edirdi,
qiymətləndirmə isə yox. Real data 2 kanallı olduğu üçün **gizli qalmışdı** —
yalnız sim (3 kanal, `p` daxil) üzə çıxardı. İki kod yolu eyni fərziyyəyə söykənir,
biri onu tətbiq etmir → tipik tələ.

**Windows Update gecə restart-ı:** 15 avqust 07:21-də `MoUsoCoreWorker.exe`
maşını yenidən başladıb (Event Id 1074). Training 03:21-də bitmişdi, əlaqəsi yoxdur.
Uzun gecə işləri üçün Windows Update "aktiv saatları" genişləndirilməlidir.

**Şəkillər:** hər model üçün `figures\report_<ad>.png` (6 panel) və
`figures\pred_<ad>_6300_20_w100.gif` (həqiqət/model/fərq) mövcuddur.

---

## 7. Plan

### Həftə 1 — infrastruktur
- [ ] Window dataloader: `.h5` → `(N,20,32,64,3)`.
      **`stride ≥ 20` MÜTLƏQDIR** — yoxsa təşkilatçıların düşdüyü sızma tələsinə özümüz düşərik.
      Train/val bölgüsü **trayektoriya səviyyəsində**, window səviyyəsində yox.
- [ ] `scoring.py`-ı lokal validation split-ə bağla (gündə 1 submission → kor atmaq olmaz)
- [ ] Baseline checkpoint-i `predict()`-ə sarıyıb **ilk submission** — məqsəd xal deyil,
      pipeline-ın işlədiyini təsdiqləmək

### Həftə 2–3 — model
- [ ] **20 kadr → 20 kadr birbaşa** (autoregressiv YOX: daha sürətli + xəta yığılmır)
- [ ] Zamanı kanala yığ: giriş 20×2=40 kanal → çıxış 40 kanal.
      Kiçik U-Net və ya FNO-2D, 32×64-də ~1–3 ms/sample hədəf
- [ ] **Sim-də pretrain → real-da finetune** (Track 1-in bütün mahiyyəti budur)
- [ ] Re/AoA verilmədiyi üçün input pəncərəsindən **rejim embedding-i** çıxaran kiçik encoder

### Həftə 4 — SPS
- [ ] Bound-lar üçün ayrıca kalibrlənmə başlığı (quantile regression və ya ensemble spread)
- [ ] Per-element `P(örtmə)·exp(−w/σ)` optimizasiyası
- [ ] Maskalı xanaların (u=v=0) SPS-də sayılmadığını nəzərə al

### Həftə 5–6 — Decision Phase hazırlığı
- [ ] **Unutma: top-10-a düşsək, təşkilatçılar training kodumuzu sıfırdan işə salacaq**
      və **görünməmiş Re/AoA-da** test edəcəklər
- [ ] → Leaderboard-a overfit etmək mənasızdır
- [ ] → Training pipeline **ilk gündən reproducible** olmalıdır
- [ ] Final paket formatı dev faza bitməmişdən ~5 gün əvvəl elan olunacaq (FAQ səhifəsində)

---

## 8. Qaydalar — diqqət ediləsi məqamlar

- **Xarici data QADAĞANDIR.** Yeni simulyasiya, xarici dataset, başqa data-da öyrədilmiş
  pretrained model — hamısı qadağan. İcazə verilən: bu release-dən öyrədilmiş checkpoint-lər,
  o cümlədən rəsmi baseline-lar.
- **Augmentation İCAZƏLİDİR** (9 avqust aydınlaşdırması) — şərt: hər training sample-ı
  release-ə qədər izlənə bilən bir transformasiya ilə bağlanmalıdır. Augmentation kodu
  training kodu ilə birlikdə təhvil verilir.
- Səbəb: final retraining mərhələsində təşkilatçıların əlində yalnız release var.
  **Release-dən reproduce olunmayan training final mərhələdə keçmir.**
- Evaluation data bütövlüyü: başqa window-ların məlumatından istifadə, konteyner
  fayl sistemindən evaluation data oxumaq, data-nı çıxışa kodlamaq → **diskvalifikasiya**.
- İcazəli: **sənə verilmiş input pəncərəsindən** interval qiymətləndirmək — SPS elə bunu mükafatlandırır.

---

## 9. Verdikt

**Qatıl.** Səbəblər:
- Leaderboard sıfırlanıb, 6+ həftə var
- Mükafat: 6000$ / 3000$ / 1500$ (hər track), top-3 NeurIPS-də oral,
  **top-5 ortaq məqalədə həmmüəllif** (+ 1 elmi rəhbər nominasiyası)
- Sahə niş: fizika + SciML → Kaggle-dakı 3000 komandalıq tabular yarışlardan qat-qat az rəqib
- **SPS-də açıq-aşkar istismar edilməmiş boşluq var**

**Track 1-lə başla, Track 2-ni sonra.** Data eynidir, Track 1-in interfeysi sadədir
(sadəcə `predict()`), Track 2 isə streaming TTT loop tələb edir. Amma **submission kvotası
tracklər arasında paylaşılır** (100/faza, gündə 1) — paralel getmək kvotanı yarıya bölür.
Track 1-də model hazır olandan sonra Track 2-yə köçürmək asandır.

---

## 10. Faydalı linklər

- Yarış saytı: https://realpdecompetition.github.io/
- Track 1 Codabench: https://www.codabench.org/competitions/17363/
- Track 2 Codabench: https://www.codabench.org/competitions/17385/
- Data (HF): https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data
- RealPDEBench kodu: https://github.com/AI4Science-WestlakeU/RealPDEBench
- Əlaqə: realpde-competition@googlegroups.com

### Baseline checkpoint ölçüləri (HF-də)
| Model | fp32 | fp16 |
|---|---|---|
| CNO | 31 MB | — |
| Transolver | 48 MB | — |
| FNO | 384 MB ❌ (limit 256) | 192 MB ✅ |

FNO-nun spektral çəkiləri `complex64`-dür → sadə `.half()` onları sındırır.
`pack_ckpt_fp16.py` onları `view_as_real(t).half()` kimi saxlayır.
Rəsmi Transolver baseline-ı ~1 dəqiqəyə işləyir (5 dəq limitə qarşı).
