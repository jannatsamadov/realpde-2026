# RealPDE — bir səhifəlik xülasə

Detallar: `NeurIPS 2026 RealPDE Competition\izah.md`. Bu fayl yalnız yenidən
işə qayıdanda 2 dəqiqəyə mənzərəni bərpa etmək üçündür.

## Tapşırıq
20 kadr sürət sahəsi ver → növbəti 20 kadrı proqnozlaşdır. `(N,20,32,64,3)`,
kanallar `[u,v,p]`, `p` real data-da yoxdur. 20 kadr = **0.4 saniyə**.
Track 1 bitir: **27 sentyabr 2026**. Mükafat: **$6k / $3k / $1.5k, hər track ayrıca.**

## Harada dayanmışıq
| | final | sıra |
|---|---|---|
| 1-ci submission (`unet_v1`) | 73.66 | 102/138 |
| **2-ci submission (`adv_v2`)** | **78.77** | ? |

Son subscore-lar: `rel_l2 94.55 · tke 74.63 · mvpe 93.49 · time 87.15 · sps 36.17`

## Beş metrikadan hansı vacibdir
`rel_l2`, `mvpe`, `time` — hamısı **87-dən yuxarı**, orada cəmi bir neçə xal qalıb.
Bütün boşluq **`tke` (74.6)** və **`sps` (36.2)**-dədir.

⚠️ **Struktur hədd:** `sps ≤ 100 × acc`, bizdə `acc = 0.69` → **SPS 69-u keçə bilməz.**
Bantları mükəmməl etsək belə `final = 88.6`. **90 üçün `tke` ~92 lazımdır.**

## Sınadıqlarımız
| | nəticə |
|---|---|
| trivial baseline-lar (persistence, time_mean) | rel_l2 ~94 — metrika yumşaqdır |
| loss-a TKE həddi (`w_tke` 0.5, 2.0) | ❌ pisləşdirdi |
| daşınma prioru **tək başına** (9 variant) | ❌ hamısı persistence-dən pis |
| daşınma şəbəkənin **girişi** kimi | ✅ 89.17 → 89.46 |
| sim pretrain → real finetune | ✅ 89.46 → **89.70** (ən yaxşı) |
| model böyütmək (base 64→128) | ❌ +0.13, 4× ölçü — dəyməz |
| **SPS bant kalibrləməsi** | ✅✅ **22.96 → 48.74** (lokal), lövhədə 16.6 → 36.2 |
| `α` amplituda düzəlişi (1.15) | ~ +0.27, kiçik |

## Modellər (273 val pəncərəsi, hamısı default bandla)
| model | parametr | rel_l2 | tke | mvpe | seçim |
|---|---|---|---|---|---|
| rəsmi FNO | 50.4M | **96.59** | **78.32** | **97.02** | **90.64** |
| **bizim `adv_finetuned`** ← göndərildi | 1.8M | 96.25 | 76.32 | 96.54 | 89.70 |
| rəsmi CNO | 8.0M | 95.96 | 75.73 | 96.25 | 89.31 |
| bizim `tke0` ← 1-ci submission | 1.8M | 96.06 | 75.37 | 96.10 | 89.17 |
| rəsmi Transolver | 12.5M | 93.85 | 70.86 | 94.40 | 86.37 |

**Rəsmi FNO bizi keçir**, amma 28× böyükdür. Bizim SPS üstünlüyümüz modeldən yox,
bant kalibrləməsindən gəlir — o texnika istənilən modelə qoşula bilər.

## Hazır, göndərilməyib
- bant miqyası tənzimlənməsi → **+0.58**
- `α = 1.15` → **+0.27**
- ikisi birlikdə: təxminən **79.6**

## Növbəti seçimlər

Hədəf: **`tke` 74.6 → ~92**, çünki `final = 90` üçün başqa yol yoxdur.
`tke` iki dəfə ödəyir — öz xalı + SPS tavanını (`100×acc`) qaldırır.

### 1. Ucuz qazancları göndər · 15 dəq · **+0.85** · ~əmin
Bant miqyası (+0.58) və `α = 1.15` (+0.27). Hazırdır, sadəcə paketləmək qalıb.
**Mənfi:** kiçikdir, bugünkü submission haqqını yeyir.

### 2. Rəsmi FNO fp16 + bizim bantlar · saatlar · **+2…5** · orta əminlik
FNO dəqiqlikdə bizi keçir (90.64 vs 89.70) → SPS tavanı da yüksəkdir.
**Müsbət:** model öyrətmək lazım deyil; checkpoint açıq buraxılıb, kit yükləyici verir.
**Mənfi:** 384 MB → fp16 (~192 MB) məcburidir; 50M parametr `time_score`-u salır;
**Decision Phase-də təşkilatçılar metodu sıfırdan öyrədir** — hazır checkpoint
üzərində qurulmuş həll orada zəif görünə bilər.

### 3. Generativ / diffusion model · günlər · **+8…12 potensial** · **qeyri-müəyyən**
`tke` 92-yə çatmağın yeganə real yolu. MSE şərti ortanı öyrədir → hamar proqnoz →
TKE ölür. Diffusion bu problemin standart cavabıdır; RealPDEBench-in `WDNO`
baseline-ı da bu ailədəndir.
**Müsbət:** yeganə yol ki, tavana aparır; Decision Phase üçün də düzgün həlldir.
**Mənfi:** bir neçə gün; `time_score` çox düşə bilər (diffusion çox addım tələb edir);
nəticə **zəmanətsizdir**.
**Qeyd:** az addımlı variant (consistency / flow-matching) sürət problemini yumşaldır.

### 4. Ehtimallı çıxış (multi-sample) · 1–2 gün · **+3…6** · orta
Model tək proqnoz yox, **paylanma** versin. Eyni anda iki metrikaya işləyir:
nümunələrin yayılması SPS bantlarını **post-hoc yox, təbii** verir, orta isə
dalğalanmanı saxlayır. Diffusion-un yüngül variantıdır.

### 5. Ensemble · saatlar · **+1…3** · orta
Bir neçə modeli (bizimki + FNO + CNO) birləşdir. Ortalama `rel_l2`-ni qaldırır,
amma **`tke`-ni AŞAĞI salır** (ortalama hamarlayır) — ehtiyatlı olmaq lazımdır.

### 6. Track 2 (LTTTA) · — · ayrı $6k · —
Ayrı mükafat, **ayrı gündəlik submission**, kod bazamızın ~80%-i köçür.
Əlavə lazım olan: `ttt_step` örtüyü + onlayn adaptasiya siyasəti.

---

## İstifadəçi ideyaları — qiymətləndirmə

### A. Fiziki dayanıqsızlıq naxışlarını training-ə əlavə etmək
*(Kelvin–Helmholtz, Rayleigh–Taylor, firehose, mirror)*

**Dəqiqləşdirmə — dörddən yalnız biri bu axında var:**
| | var? | səbəb |
|---|---|---|
| **Kelvin–Helmholtz** | ✅ **üstündür** | profilin shear təbəqələri məhz KH ilə burulğana çevrilir |
| Rayleigh–Taylor | ❌ | sıxlıq təbəqələnməsi + təcil tələb edir; təkfazalı suda yoxdur |
| Firehose | ❌ | **plazma** dayanıqsızlığı (anizotrop təzyiq + maqnit sahə) |
| Mirror | ❌ | eyni — plazma fizikası |

**Müsbət:** fikrin nüvəsi doğrudur — modelə fiziki quruluşu öyrətmək.
**Mənfi:** bizim data **onsuz da KH ilə doludur** (100 rejim, 100k kadr). Kənar KH
simulyasiyaları hədəfdən **daha uzaq** olardı, halbuki eyni konfiqurasiyalı sim
data-mız cəmi **+0.25** verdi.
**Diaqnoz dəstəkləmir:** model naxışı **tanıyır** — dalğalanma korrelyasiyası 0.63,
TKE xəritəsi düzgün yerlərdə. Uğursuzluq **fazadadır**: korrelyasiya 20 kadrda
0.86 → 0.55 çürüyür. Bu, xaosdur, tanınma problemi deyil.
**Gözlənilən:** **~0**. Absurd deyil, sadəcə problemimizə dəymir.
**İşləyə bilən variantı:** KH-i əlavə data kimi yox, **köməkçi tapşırıq** kimi —
model eyni anda burulma/shear təbəqəsinin yerini proqnozlaşdırsın. Ucuzdur (saatlar),
gözlənilən **+0…1**.

### B. Çoxlu simulyasiyada pretrain
**Artıq edilir.** sim = **100 fərqli iş rejimi** (20 Re × 5 AoA), 100 000 kadr,
hamısında pretrain olunur (`sim_pretrain` → `adv_finetuned`). Qazanc **+0.25**.

**Genişləndirilmiş variantı:** başqa PDE datasetlərində pretrain (PDEBench, PDEArena) —
"PDE foundation model" yanaşması. RealPDEBench-in `DPOT` baseline-ı məhz budur.
**Müsbət:** qat-qat çox data; ümumi operator quruluşu öyrənilə bilər.
**Mənfi:** başqa həndəsə, sərhəd şərtləri, ayırdetmə; onlarla GB endirmə;
**eyni konfiqurasiyalı sim cəmi +0.25 verdisə, uzaq data daha az verər.**
**Gözlənilən:** **+0…1**, xərci yüksək.

## Əmrlər
```powershell
cd D:\Projects\Competitions\realpde
$py = ".\.venv\Scripts\python.exe"

& $py scripts\compare_runs.py                                   # bütün modellər
& $py scripts\model_report.py --checkpoint checkpoints\adv_finetuned_best.pt
& $py scripts\calibrate_sps.py --checkpoint checkpoints\adv_finetuned_best.pt
& $py scripts\build_submission.py --checkpoint checkpoints\adv_finetuned_best.pt --tag v3 --bounds
& $py scripts\what_would_it_take.py                             # 90 üçün nə lazımdır
```

## Diqqət
- **GPU fanı işləmir** → yük 72%, 78°C-də dayan (`train.py` bunu özü edir)
- Gündə **1 submission** — paketləməni həmişə `build_submission.py` ilə yoxla
  (bir dəfə səhv arxitektura tutuldu, göndərilsəydi bütün xallar sıfır olardı)
- Lokal xallar **nikbindir**: gizli setdə rel_l2/tke/mvpe −1.7…−3.0, **sps −12.6**
- Repo: https://github.com/jannatsamadov/realpde-2026 (private)
