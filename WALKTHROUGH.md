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
1. **Ucuz qazancları göndər** (15 dəq) → ~79.6
2. **Rəsmi FNO fp16 + bizim bantlar** (bir neçə saat) → ehtimalla 81–84.
   Qeyd: Decision Phase-də təşkilatçılar metodu sıfırdan öyrədir, hazır
   checkpoint üzərində qurulmuş həll orada zəif görünə bilər.
3. **Track 2** (LTTTA) — ayrı mükafat, ayrı gündəlik submission, kodun 80%-i köçür.
4. **Generativ model** — `tke` 92-yə çatmağın yeganə real yolu, amma bir neçə gün.

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
