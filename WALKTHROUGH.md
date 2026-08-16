# RealPDE Track 1 — vəziyyət

Detallar: `NeurIPS 2026 RealPDE Competition\izah.md`. Bu fayl işə qayıdanda
2 dəqiqəyə mənzərəni bərpa etmək üçündür. Son yenilənmə: 17 avqust 2026.

## Tapşırıq
20 kadr sürət sahəsi → növbəti 20 kadr. `(N,20,32,64,3)`, kanallar `[u,v,p]`.
`p` real datada ölçülmür. **20 kadr = 0.4 saniyə** (dt = 0.02 s).
Ölçmə **2D2C**: müstəvi kəsik, üç sürət komponentindən yalnız ikisi. `w` heç yerdə yoxdur.

**Track 1-də ardıcıllıq YOXDUR** — hər pəncərə müstəqildir, yaddaş yoxdur.
Ardıcıl rejim Track 2-nin mövzusudur.

Dev Phase bitir: **27 sentyabr 2026**. Mükafat: **$6k / $3k / $1.5k, hər track ayrıca.**
Kvota: gündə 1 submission (hər track ayrı), faza üzrə 100 (paylaşılır).

## Leaderboard
| submission | rel_l2 | tke | mvpe | time | sps | **final** |
|---|---|---|---|---|---|---|
| `unet_v1` | 94.635 | 73.677 | 93.160 | 92.303 | 16.644 | 73.665 (102/138) |
| **`adv_v2`** ← ƏN YAXŞI | 94.547 | 74.631 | 93.487 | 87.153 | 36.170 | **78.766** |
| `adv_v3` | 94.424 | 74.802 | 93.487 | 86.916 | 36.334 | 78.744 |

Qazancın **~90%-i SPS bant kalibrləməsindən** gəldi (≈+4.6), modeldən yox (≈+0.5).

## Model: `checkpoints\adv_finetuned_best.pt`
Advective U-Net, **1.8M parametr**, sim pretrain → real finetune.
- zaman kanala yığılır: 40 → 40 kanal, **birbaşa** proqnoz (autoregressiv yox)
- **residual**: son giriş kadrından fərq; sıfır-init başlıq → persistence-dən başlayır
- girişdə: daşınma prioru (`advect_sequence`, fırlanma daxil), burulma, divergensiya,
  `|V|`, mütləq koordinatlar
- **Re və AoA HEÇ VAXT verilmir** — `metadata` inference-də boşdur

## Nə işlədi, nə işləmədi
| | nəticə |
|---|---|
| **SPS bant kalibrləməsi** | ✅✅ **16.6 → 36.2** lövhədə |
| sim pretrain → real finetune | ✅ +0.25 |
| daşınma şəbəkənin **girişi** kimi | ✅ +0.29 |
| daşınma **hazır cavab** kimi (9 variant) | ❌ hamısı persistence-dən pis |
| TKE həddi loss-da (w 0.5, 2.0) | ❌ monoton pisləşdirdi |
| model böyütmək (base 64→128) | ❌ +0.13, 4× ölçü |
| post-hoc `α` (1.15) | ❌ lokalda +1.2, lövhədə **+0.17** |
| bantları 1.49× genişlətmək | ❌ lokalda +1.9 proqnoz, lövhədə **+0.16** |
| **generativ flow matching** (2 dizayn) | ❌ heç bir metrikada üstün deyil |
| hər-xana deformasiya adaptasiyası | ❌ 48.67 → 43.83 |
| pəncərə üzrə orta deformasiya (skalyar) | ~ +0.9 lokal, sınanmayıb |

**Ən böyük dərs:** öz split-imizdə tənzimlənən post-hoc düzəlişlər gizli setə **köçmür**
(iki müstəqil təcrübə). Struktur dəyişikliklər köçür.

## Modellərin müqayisəsi (273 val pəncərəsi, default band)
| model | parametr | rel_l2 | tke | mvpe | seçim |
|---|---|---|---|---|---|
| rəsmi FNO | 50.4M | **96.59** | **78.32** | **97.02** | **90.64** |
| **bizim `adv_finetuned`** | **1.8M** | 96.25 | 76.32 | 96.54 | 89.70 |
| rəsmi CNO | 8.0M | 95.96 | 75.73 | 96.25 | 89.31 |
| rəsmi Transolver | 12.5M | 93.85 | 70.86 | 94.40 | 86.37 |

Rəsmi FNO bizi keçir, amma **28× böyükdür**. Checkpoint-lər `baselines\`-dədir
(gitignore). Onlar **normallaşdırılmış fəzada** işləyir — xam m/s vermək rel_l2
xətasını 2.37 edir.

## Ölçülmüş fiziki hədlər
- **PIV şumu** hədəf dispersiyasının **~22%-i** (lag-1 korrelyasiya ilə).
  Şumsuz sim-də model tke 80.32 alır, şumlu real-da 76.32 → şum ≈ **4 xal**.
- **Şum ∝ deformasiya**: ən yüksək deformasiya kvintilində şum ən aşağıdan **9.9×**.
  Amma `sigma` xəritəsi bunu onsuz da daxil edir → əlavə etmək zərər verir.
- **Xaos**: tökülmə tezliyi 3.08 Hz → dövr 0.325 s. Üfüq 0.4 s = **1.23 dövr**.
  Dalğalanma korrelyasiyası 20 kadrda **0.86 → 0.55**. Bu, modelin qüsuru deyil.
- **3D**: `∂w/∂z = −(müstəvi divergensiyası)`. Sim-də div/vort = **0.33** → axın
  həqiqətən 3D-dir. AoA ilə cəmi **+12%** artır → rejim fərqlərini izah etmir.
  ⚠️ Real datada bu ölçmə **əks işarə** verir (şum artefaktı) — fizika sualları
  üçün **sim işlətmək lazımdır**.
- **Q-kriteriyası**: ölçülən xanaların yalnız **14.6%-i** fırlanma-dominantdır.
  Vorticity-də qalın görünən shear təbəqələri əslində **deformasiya**dır.
- **Vorteks birləşməsi YOXDUR** — tezlik 3.08 Hz-də sabit. St = 0.202 (klassik 0.2).

## Rejimlərə görə davranış
| | rel_l2 | tke |
|---|---|---|
| Re 3750, AoA 0° | 97.40 | **86.40** |
| Re 26761, AoA 0° | 97.33 | **67.91** |
| Re 26761, AoA 15° | 93.88 | 78.37 |

`tke` çətinliyi **Re ilə monoton deyil** — AoA 0°-də həqiqi dalğalanma zəifdir və
şumla örtülür, ona görə orada TKE ümidsizdir.

## Struktur məhdudiyyət — 90-a çatmaq üçün
```
sps = 100 × acc × Q,   Q ≤ 1
acc = 0.5(1−pm_rel_l2) + 0.3(1−pm_tke) + 0.2(1−pm_mvpe)
```
→ **`sps ≤ 100 × acc`**. Bizdə `acc = 0.69` → **SPS 69-u keçə bilməz**.
Bantları mükəmməl etsək `final = 88.6`. **90 üçün** təxminən
`rel_l2 97 · tke 88 · mvpe 97 · time 93 · sps 60` lazımdır.
`tke` tavanı şuma görə ~90 civarındadır.

`final_score` düsturu **açıqlanmayıb**. İki submission-dan uyğun bir həll:
`w_sps ≈ 0.30`, digərləri ≈ 0.19 — yəni SPS bərabər paydan artıq çəkidədir.

## Növbəti seçimlər
1. **Track 2 (LTTTA)** — ayrı $6k, ayrı gündəlik submission, kodun ~80%-i köçür.
   Əlavə: `ttt_step` örtüyü + onlayn adaptasiya siyasəti.
2. **Rəsmi FNO fp16 + bizim bantlar** — FNO dəqiqlikdə üstündür, bantlar istənilən
   modelə qoşulur. Maneə: 384 MB → fp16 (~192 MB); `time_score` düşür;
   Decision Phase-də metod sıfırdan öyrədilir.
3. **Köməkçi hədəf: Re/AoA proqnozu** — girişə vermək olmaz (metadata boşdur),
   amma **əlavə çıxış** kimi öyrətmək olar. Gözlənilən: kiçik (model rejimi onsuz
   da 0.969 korrelyasiya ilə oxuyur). ~30 dəq.
4. **Epoxa artırmaq** — əyri yastılanıb (+0.045 son addımda), ~+0.2 gözlənilir.

## Əmrlər
```powershell
cd D:\Projects\Competitions\realpde
$py = ".\.venv\Scripts\python.exe"

& $py scripts\compare_runs.py                    # bütün modellər
& $py scripts\model_report.py --checkpoint checkpoints\adv_finetuned_best.pt
& $py scripts\calibrate_sps.py --checkpoint checkpoints\adv_finetuned_best.pt
& $py scripts\build_submission.py --checkpoint checkpoints\adv_finetuned_best.pt --tag v4 --bounds
& $py scripts\what_would_it_take.py              # 90 üçün nə lazımdır
& $py scripts\regime_gif.py --case 26700_15      # real/model/fərq/sim GIF
& $py scripts\sim_regime_report.py               # 5 rejim üzrə
& $py scripts\vortex_structure.py --case 20325_15
& $py scripts\noise_vs_strain.py
& $py scripts\out_of_plane.py
```

## Diqqət
- **GPU fanı işləmir** → yük 72%, 78°C-də dayan. İki GPU işi eyni anda **heç vaxt**.
- Gündə **1 submission** — həmişə `build_submission.py` ilə yoxla. Bir dəfə səhv
  arxitektura tutuldu; göndərilsəydi bütün xallar sıfır olardı.
- Lokal xallar **nikbindir**: gizli setdə rel_l2/tke/mvpe −1.7…−3.0, **sps −12.6**.
- Val split-inin daxili yayılması **3-4 xaldır** → ondan kiçik fərq sübut deyil.
- `izah.md` və bu fayl **şəxsi qeydlərdir**, repo məzmunu deyil — sonra
  `.gitignore`-a salınacaq.
- Repo: https://github.com/jannatsamadov/realpde-2026 (private)
