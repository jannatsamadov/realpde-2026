kevin helmholtz və bənzər maye dayanıqsızlıqlarını hansı ai modellər ilə rahat öyrənmək olar? bunun üçün hansı metodlar var ki prosesi sürətləndirsin və dəqiqliyi artırsın? məqalələr və elmi araşdırmalar, öz bilik və təcrübən bunlardan çıxış edib ciddi cavab verməyni gözləyirəm

Я вас не понимаю. Я просто языковая модель и не смогу вам с этим помочь.

kevin helmholtz və bənzər maye dayanıqsızlıqlarını hansı ai modellər ilə rahat öyrənmək olar? bunun üçün hansı metodlar var ki prosesi sürətləndirsin və dəqiqliyi artırsın? məqalələr və elmi araşdırmalar, öz bilik və təcrübən bunlardan çıxış edib ciddi cavab verməyni gözləyirəm

Kelvin-Helmholtz (KH) dayanıqsızlığı, xüsusən də fərqli sürətlərə malik iki axın arasındakı kəsilməzlik sərhədində yaranan mürəkkəb qeyri-xətti dinamikanı əhatə edir. Bu cür maye və plazma dayanıqsızlıqlarının (həmçinin Rayleigh-Taylor dayanıqsızlığı) öyrənilməsi üçün ənənəvi Hesablamalı Maye Dinamikası (CFD) metodları hesablama baxımından çox ağır ola bilər. Süni intellekt modelləri bu prosesi həm sürətləndirmək, həm də analitik dəqiqliyi artırmaq üçün fəal şəkildə istifadə olunur.

Aşağıda bu məqsədlə istifadə olunan ən effektiv AI modelləri, prosesi optimallaşdıran metodlar və elmi yanaşmalar qeyd edilmişdir.

Maye Dayanıqsızlıqları Üçün Ən Uyğun AI Modelləri
Fizikadan Məlumatlı Neyron Şəbəkələri (PINNs - Physics-Informed Neural Networks): Maye dinamikası üçün ən fundamental AI yanaşmasıdır. PINN-lər Navier-Stokes, istilik keçiriciliyi və ya Maqnitohidrodinamika (MHD) tənliklərini birbaşa itki funksiyasına (loss function) daxil edir. Bu, modelin sadəcə dataya "uyğunlaşmasının" qarşısını alır və qorunma qanunlarına (kütlə, impuls, enerji) riayət etməsini təmin edir. Xüsusilə plazma dinamikası və mürəkkəb sərhəd şərtləri olan sistemlərdə yüksək dəqiqlik verir.

Latent Neyron Diferensial Tənlikləri (Latent Neural ODEs): KH dayanıqsızlığı zamanla davamlı təkamül edən dinamik sistemdir. Latent ODE modelləri sistemin zaman sıraları datası (məsələn, axının fərqli anlardakı vəziyyəti) üzərindən çıxarış edərək, latent fəzada dinamikanı kəsilməz bir diferensial tənlik kimi modelləşdirir. Bu, standart rekurrent şəbəkələrdən (RNN) fərqli olaraq, zaman addımlarından asılılığı aradan qaldırır və daha fiziki, davamlı bir həll təqdim edir.

Fourier Neyron Operatorları (FNO - Fourier Neural Operators): FNO-lar ənənəvi şəbəkələrdən fərqli olaraq, sonlu ölçülü vektorlar arasında deyil, sonsuz ölçülü funksiya fəzaları arasında xəritələmə aparır. Həlləri Fourier fəzasında (tezlik domenində) öyrəndiyi üçün, müxtəlif həndəsi torlardan (mesh-independent) asılı olmayaraq işləyə bilir və qeyri-xətti burulğanlı (turbulent) axınlarda ənənəvi CFD həllədicilərindən minlərlə dəfə daha sürətli nəticə verir.

Sürəti və Dəqiqliyi Artıran Metodologiyalar
Dinamik sistemlərdə sadəcə modeli seçmək kifayət etmir, öyrənmə prosesində riyazi arxitekturanın optimallaşdırılması həlledici rol oynayır:

Sərt Diferensial Həllədicilərin (Stiff Solvers) İnteqrasiyası: KH dayanıqsızlığının inkişaf fazasında (burulğanların yaranması və qırılması anında) qradiyentlər kəskin şəkildə dəyişir. Bu cür dinamikanı Latent ODE və ya analoji modellərlə öyrədərkən standart həllədicilər (məsələn, Runge-Kutta 4) ədədi dayanıqsızlığa səbəb olur. Neyron şəbəkəsinin arxa fonunda Rosenbrock ailəsinə aid (məsələn, Rosenbrock23) və ya digər qapalı (implicit) sərt həllədicilərdən istifadə etmək həm addım sayını azaldaraq prosesi sürətləndirir, həm də hesablamanın partlamasının qarşısını alır.

Çoxdəqiqlikli (Multi-fidelity) Təlim və Boussinesq Yaxınlaşması: Tam qeyri-xətti sıxılan maye və ya plazma tənlikləri üzərində modeli sıfırdan öyrətmək çətindir. Təlimi sürətləndirmək üçün "Transfer Learning" yanaşmasından istifadə edilir. Model əvvəlcə aşağı rezolyusiyalı məlumatlar və ya Boussinesq yaxınlaşması kimi sadələşdirilmiş termodinamik konveksiya ssenariləri üzərində öyrədilir, daha sonra yüksək dəqiqlikli datalar üzərində "fine-tune" edilir.

Diferensiallana bilən Fizika (Differentiable Physics): Ənənəvi ədədi həllədicilərin Pytorch və ya JAX kimi diferensiallana bilən mühitlərdə yenidən yazılması metodudur. Bu zaman tam simulyasiyanı neyron şəbəkəsi ilə əvəz etmək əvəzinə, AI yalnız həllədicinin zəif qaldığı (və ya ən çox vaxt aparan) hissələri – məsələn, kiçik miqyaslı turbulentlik effektlərini hesablayır.

Elmi İstinadlar və Ədəbiyyat İstiqamətləri
Bu sahədə tədqiqat apararkən aşağıdakı fundamental məqalələrə və müəlliflərə istinad etmək tövsiyə olunur:

M. Raissi, P. Perdikaris, G.E. Karniadakis (2019): Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. PINN memarlığının əsasını qoyan və maye dinamikasına tətbiqini göstərən təməl iş.

Z. Li et al. (2020): Fourier Neural Operator for Parametric Partial Differential Equations. FNO-nun Navier-Stokes tənliklərində ənənəvi həllədiciləri necə üstələdiyini göstərən tədqiqat.

R. T. Q. Chen et al. (2018): Neural Ordinary Differential Equations. Latent ODE yanaşmasının əsasını qoyan və kəsilməz zamanlı sistemlərin AI ilə inteqrasiyasını sübut edən iş.

Xüsusilə heliofizika və günəş küləyi şərtlərində plazma dayanıqsızlıqlarını tədqiq edən müasir ədəbiyyatda, DeepONet və Physics-Informed Neural Operators (PINO) modellərinin həndəsi deformasiyalara və MHD tənliklərinə uyğunlaşdırıldığı məqalələr ən aktual istinad nöqtələridir. Python ekosistemində SciML (və ya JAX/Flax mühitləri) bu tədqiqatları reallaşdırmaq üçün ən güclü alətlərdir.

# One-shot prompt for an independent design opinion



Paste everything below the line into a fresh session of another model.



**What is deliberately NOT in it:** our architecture, our training recipe, our

interval strategy, every result we have measured, and our leaderboard scores.

Including any of those turns an independent design into a copy of ours and

destroys the point. Everything that IS in it is either published by the

organizers or directly verifiable from the released data.



The competition is not named on purpose — naming it invites retrieval of the

official baselines and of other people's write-ups instead of a fresh design.



---



You are designing a machine-learning system for a scientific forecasting

benchmark. You do not have access to the data. Work from the specification

below, which is complete and authoritative, plus your own knowledge of fluid

dynamics and ML. Commit to one concrete design; do not give me a survey of

options.



## The task



Given 20 consecutive frames of a two-dimensional velocity field, predict the

next 20 frames.



- Tensor shape in and out: `(N, 20, 32, 64, 3)`, float32, channels ordered

  `[u, v, p]`.

- `u, v` are the two in-plane velocity components in m/s. `p` is pressure.

- Frame interval `dt = 0.02 s`, so each window spans 0.4 s and the forecast

  horizon is another 0.4 s.

- Uniform grid, `dx = dy = 3.422 mm`, i.e. a field of view of about

  21.9 cm x 10.8 cm.



## The physical setting



Flow past a stationary airfoil in a water tunnel. Chord length is approximately

7.2 cm. The airfoil is at a fixed location in the frame in every sample and

never moves.



Two regime parameters vary across the dataset: Reynolds number and angle of

attack.



**The measurement is a planar slice of a three-dimensional flow, and it is

2D2C**: only two of the three velocity components are recorded, on one plane.

The out-of-plane component is not available anywhere in the data.



## The two data splits



**Simulated split.** 100 trajectories, 1000 frames each. 20 distinct Reynolds

numbers from 3750 to 27975, crossed with 5 angles of attack (0, 5, 10, 15, 20

degrees). All three channels `u, v, p` are present and the fields are clean.

The simulation is non-dimensionalised.



**Real split.** 81 trajectories, 282 to 868 frames each, 69085 frames in total.

18 distinct Reynolds numbers from 3750 to 26700, same 5 angles of attack (the

grid is not complete; some combinations are missing). Measured by Particle Image

Velocimetry, so the fields carry measurement noise. Only `u, v` are measured;

`p` is not measured and is stored as zeros. Values are in m/s.



Cells outside the illuminated field of view, and cells inside the airfoil body,

are stored as exact zeros in the real split.



**Scoring is on real, held-out trajectories at Reynolds numbers not present in

training.**



## What the model is given at inference



An input window, and nothing else. **Reynolds number and angle of attack are NOT

provided at scoring time** — the metadata field is empty on scored calls.

Anything the model needs to know about the regime must be inferred from the

input window itself.



## The metrics



Five subscores, each in [0, 100]. Higher is better.



Three accuracy subscores, computed on the measured channels `u, v` only

(`p` is never scored), per window and then averaged:



```

score = 100 / (1 + 0.5 * error)

```



- **rel_l2**: `||pred - target|| / ||target||` over the u, v field.

- **tke**: relative L2 of the turbulent kinetic energy field

  `0.5 * (var_t(u) + var_t(v))`, where the variance is over the 20 frames of

  the window. Note this is a relative L2 of the TKE *map*, not of a scalar.

- **mvpe**: relative L2 of the time-averaged `u, v` at a fixed grid of 36 probe

  points in the wake behind the airfoil.



One speed subscore, where `t_neural` is the mean per-sample inference wall time

in seconds:



```

time_score = 100 / (1 + sqrt(t_neural / 0.72896))

```



The constant 0.72896 s is the runtime of the numerical solver this benchmark is

positioned against.



One calibration subscore, **SPS**. The model may optionally return a lower and

an upper bound per element, the same shape as the prediction. Per element:



```

nil     = (upper - lower) / sigma_global

element = (1 - pm) * exp(-nil)     if lower <= target <= upper, else 0

```



with `sigma_global = 0.0563870259` frozen for the season. `pm = e / (0.5 + e)`

squashes each of the three accuracy errors `e`, and the three resulting branches

are combined with weights DM 0.5 / TKE 0.3 / MVPE 0.2, then multiplied by 100.

Elements outside the field of view or inside the airfoil leave the average

entirely, numerator and denominator both.



Bounds are all-or-nothing across a run: supply them on every call or on none. If

you supply none, the scorer substitutes a default band of +-5% of |prediction|.



The five subscores are combined into a single `final_score` by a rule the

organizers have **not published**.



## Hard constraints



- Inference runs offline in a container with PyTorch 2.2.2, CUDA 12.1,

  Python 3.10, numpy. **scipy, h5py, pandas, sklearn, einops and OpenCV are NOT

  installed** and nothing can be installed at evaluation time. Pure-Python

  dependencies must be vendored into the submission archive.

- The extracted submission archive, weights included, must stay under 256 MB.

- No network access at evaluation time.

- Whole-submission wall clock limit: 5 minutes.

- **External data is banned.** No extra simulations, no external datasets, no

  pretrained weights from any other source. Shortlisted methods are re-trained

  from scratch by the organizers, who hold only the released training data, so

  any recipe that cannot be reproduced from these two splits alone is

  disqualified. Augmentation of the released data is explicitly allowed.



## Training budget



One laptop RTX 2060, 6 GB VRAM, and its cooling is degraded so sustained load is

capped at roughly 70% utilisation. Assume a few hours of training per experiment,

not days, and no multi-GPU.



## What I want back



1. **One architecture**, specified concretely enough to implement: inputs,

   parameter count, how the 20 output frames are produced, and why that choice

   over the obvious alternatives.

2. **The training recipe**: loss, how you use the two splits, augmentation,

   schedule.

3. **Your strategy for the interval bounds**, including how you would choose

   their width. Show the reasoning, not just a rule.

4. **The first three measurements you would run on the data before writing any

   model code**, and what each one would change about your design depending on

   how it comes out.

5. **Predicted subscores** with an honest uncertainty range, and which single

   subscore you would attack first for the largest gain in the combined score.

6. **The two biggest risks** in your design and the cheapest experiment that

   would falsify each.



Where the specification leaves something ambiguous, state your assumption

explicitly and proceed rather than asking me. Be concrete and quantitative.

Flag anything you are guessing at. 

1. Architecture
Design: Direct-Time 2D U-Net (DT-UNet)
Parameter Count: ~4.8 million.

Inputs & Outputs:
The input tensor (N, 20, 32, 64, 3) is permuted and reshaped to fuse time and variables into the channel dimension: (N, 60, 32, 64).
The output tensor has shape (N, 80, 32, 64). This is reshaped back to yield two components:

Predictions: 20 frames of u, v channels (40 channels).

Calibration Widths: 20 frames of w_u, w_v bounds (40 channels).
Note: Pressure p is discarded at the output layer to save compute, as it is never scored.

Structure:

Encoder: 4 stages of standard 2D convolutions (Conv2d -> BatchNorm -> LeakyReLU). Channel progression: 64 → 128 → 256 → 512. Max-pooling (2×2) between stages.

Bottleneck: 2 layers of 512 channels. Implicitly, the model infers the unprovided regime parameters (Reynolds number and Angle of Attack) here by encoding the freestream velocity magnitude and the boundary layer separation visible in the input frames.

Decoder: 4 stages of bilinear upsampling followed by 2D convolutions, with skip connections (concat) from the encoder. Channel progression: 256 → 128 → 64 → 64.

Head: A final 1×1 convolution projecting the 64 features to 80 channels.

Why this choice:
An autoregressive model (like ConvLSTM) requires 20 sequential forward passes, accumulating error and drastically lowering the time_score. 3D Convolutions (x,y,t) are too memory-intensive for a 6GB VRAM budget and slow to train. By mapping time strictly to the channel dimension, the network treats the spatio-temporal forecasting as a single, highly optimized 2D image-to-image translation task. A ~4.8M parameter 2D U-Net takes < 0.015 seconds per inference pass, nearly maxing out the benchmark's speed metric, while easily fitting the 256 MB archive limit (occupying ~19 MB).

2. The Training Recipe
Data Strategy:
The domain gap between the Simulated (clean, non-dimensional, has p) and Real (PIV noise, dimensional m/s, p=0) splits is lethal.

Scaling: I will multiply the Simulated velocity fields by the characteristic velocity derived from the Reynolds numbers and the 7.2 cm chord length to dimensionalize them to m/s.

Channel Masking: I will forcefully zero-out the p channel in the Simulated split with a 75% probability during pretraining. This forces the network to rely on u, v dynamics rather than leaning on pressure gradients it will not have access to at inference.

Loss Function:
A composite loss mapping directly to the benchmark metrics:

L=L 
MSE
​
 (u,v)+λ 
1
​
 L 
TKE
​
 +λ 
2
​
 L 
MVPE
​
 
L 
MSE
​
  is the standard L2 loss on the velocity fields.

L 
TKE
​
  computes the temporal variance of the predicted 20 frames and penalizes the L2 distance against the target temporal variance.

L 
MVPE
​
  computes the mean over time of the prediction and penalizes the L2 distance against the target mean.
Weights: λ 
1
​
 =2.0 (to combat the U-Net's tendency to over-smooth) and λ 
2
​
 =0.5.

Augmentation:
No spatial transformations, as the airfoil is fixed and grid-aligned. I will apply additive Gaussian noise (μ=0,σ=empirical PIV variance) to the Simulated split to simulate measurement noise.

Schedule:

Phase 1 (Simulated Data): 1.5 hours. AdamW, Batch Size 16, Base LR 10 
−3
  with Cosine Annealing.

Phase 2 (Real Data Fine-tuning): 1.5 hours. Strictly on the Real split. Batch Size 16, LR dropped to 10 
−4
 .

3. Strategy for Interval Bounds (SPS)
Method:
I will predict the interval width dynamically rather than using a static rule. The final 40 channels of the output represent the bound widths W. I enforce positivity using a Softplus activation: W=Softplus(out 
40...80
​
 ).
The upper and lower bounds are constructed symmetrically:

upper=pred+W
lower=pred−W
Choosing the Width (Loss Design):
The SPS metric zeroes out if the target falls outside the bounds, making coverage the absolute priority, but penalizes wide bands exponentially via exp(−nil). To optimize this without differentiating the exact SPS logic, I will train the W channels using a heavily skewed Pinball (Quantile) Loss targeting the 95th percentile, combined with an L1 penalty on W:

L 
bounds
​
 = 
N
1
​
 ∑(max(target−upper,0)+max(lower−target,0))×50+0.1×mean(W)
This asymmetrical loss punishes the model 500x harder for failing to enclose the true value than it does for expanding the band, naturally tuning W to the edge of the empirical uncertainty while maintaining a slight pressure to keep it narrow.

4. First Three Measurements Before Coding
Sim-to-Real Kinematic Scale Factor:
Measurement: Calculate the max and mean velocities of the Simulated split versus a trajectory in the Real split at the exact same Reynolds number and Angle of Attack.
Impact: If the relationship is not perfectly linear (due to 3D vortex stretching uncaptured in the 2D simulation), I will abandon Sim-to-Real transfer learning entirely. I would change the design to train exclusively on the Real split rather than poisoning the model with physically mismatched dynamics.

Persistence Baseline Metrics:
Measurement: Run a trivial baseline where the prediction is just copying the last input frame 20 times (pred 
t
​
 =input 
t=0
​
 ), and score it through the official metrics.
Impact: If the mvpe (mean wake) score for persistence is already > 90/100, I will completely remove the L 
MVPE
​
  component from my training loss, dedicating all network capacity and gradient updates to the much harder L 
TKE
​
 .

Spatial Variance Heatmap of the Real Split:
Measurement: Compute the variance of u,v across time and average it across all Real trajectories to create a 2D spatial heatmap.
Impact: This will show exactly where the turbulence (wake) occurs. If the turbulence occupies less than 20% of the 32×64 grid, I will introduce a spatial weighting matrix to the loss function, multiplying errors in the wake region by 5.0 and ignoring the laminar freestream.

5. Predicted Subscores & Attack Plan
time_score: 93 ± 2. (Inference time for this model is ~0.01s. 100/(1+ 
0.01/0.728

​
 )≈93).

rel_l2: 78 ± 4.

tke: 52 ± 8. (Highly sensitive to the U-Net blurring out high-frequency spatial features).

mvpe: 86 ± 3.

SPS: 45 ± 10. (Calibration on out-of-distribution Reynolds numbers is notoriously unstable).

Primary Attack Target: tke (Turbulent Kinetic Energy).
Because standard MSE loss naturally regresses to the mean, it destroys predicted variance (TKE). TKE is not only a standalone accuracy score but also carries a heavy 0.3 weight in the SPS calibration score. Fixing TKE by weighting it heavily in the custom loss function will yield the highest simultaneous lift across both accuracy and calibration axes.

6. The Two Biggest Risks & Falsification
Risk 1: Time-as-channels blurs temporal progression.
Smashing 20 frames into the channel dimension assumes the network can resolve complex temporal advection purely through deep 1x1 combinations of spatial filters. It risks outputting 20 frames that look like a smeared average rather than a crisp, moving fluid.

Cheapest Falsification: Train a tiny 1-layer Convolutional GRU (autoregressive) on 5 trajectories for 20 epochs. Compare the temporal continuity (frame-to-frame L2 delta) against the DT-UNet. If the ConvGRU resolves the wake visibly better without hitting the 5-minute wall-clock limit, the direct-time design must be scrapped.

Risk 2: Out-of-Distribution (OOD) Reynolds Failure.
The model must infer the unseen Re/AoA purely from the input flow field. Since scoring is strictly on Reynolds numbers not seen in training, the U-Net may overfit to the exact boundary layer thicknesses of the training Re numbers and fail to interpolate the physics.

Cheapest Falsification: Execute a Leave-One-Out Cross Validation (LOOCV) on the Real split. Exclude the median Reynolds number from training, train for 30 epochs, and score strictly on that held-out Re. If the rel_l2 collapses by > 20% compared to the training Re scores, the architecture is merely memorizing regimes. I would then be forced to inject instance normalization layers to strip scale-dependent magnitudes and force the model to learn scale-invariant vortex dynamics.



===========================

Qwen's answer

Below is one committed design, not a menu. I state my assumptions where the spec is silent, then give the architecture, training recipe, interval strategy, first measurements, score estimates, and risks.

---

## Assumptions I am making explicitly

1. **Training-time metadata exists.** Reynolds number and angle attack are available for training, but never at scored inference. I use them only for dimensionalising the simulated data and as a weak auxiliary training signal; the scored model consumes only the input window.
2. **Simulated velocity is nondimensional by a freestream scale.** I dimensionalise simulated velocity using  
   \[
   U_\infty = \frac{Re\,\nu}{c},
   \]
   with chord \(c \approx 0.072\ \mathrm{m}\) and water kinematic viscosity \(\nu \approx 1.0\times 10^{-6}\ \mathrm{m^2/s}\). This is a guess; if the released simulated fields are already dimensional, or if this scaling is inconsistent with real velocity magnitudes, I fall back to per-trajectory robust velocity scaling matched to the real split.
3. **Pressure is not useful at scoring time.** Real pressure is zero, pressure is never scored, and using simulated pressure as an input creates a sim-to-real gap. I ignore input pressure and output zero pressure. If the scorer requires a pressure bound, I supply a negligible bound around zero but I do not rely on pressure contributing to SPS.
4. **The 36 MVPE probe coordinates may not be released.** If they are released, I use them exactly. If not, I approximate MVPE during training with a downstream wake ROI mask. I state this because the prompt gives the existence of probes but not their coordinates.
5. **The final-score aggregation is unknown.** I therefore optimize all five subscores, but I make one explicit prioritization later.

---

# 1. Architecture: scale-conditioned residual U-Net, one-shot 20-to-20 forecast

I commit to a **direct 20-frame-to-20-frame convolutional forecast network**. It produces all 20 future frames in one forward pass. No autoregressive rollout.

Name: **Scale-Conditioned Residual U-Net**, SCF-UNet.

## Input processing

Given input tensor

\[
x \in \mathbb{R}^{B\times 20\times 32\times 64\times 3},
\]

I take only channels 0 and 1, \(u\) and \(v\). Channel 2, pressure, is discarded.

### Valid mask

I construct a static valid mask from the real training data:

\[
M_{\text{static}}(y,x)=1
\]

for cells that are ever nonzero in the real training trajectories, and 0 for cells outside the illuminated FOV or inside the airfoil. At inference I intersect this with a dynamic mask

\[
M_{\text{dyn}}(y,x)=1
\quad\text{if}\quad
\max_t \sqrt{u_t^2+v_t^2} > 10^{-4},
\]

otherwise 0. The final mask is

\[
M = M_{\text{static}} \land M_{\text{dyn}}.
\]

The mask is supplied to the network as one static input channel.

### Per-sample velocity scale

The model must infer Reynolds regime from the window itself. A large part of that is the velocity scale. For each sample I compute

\[
s = \max\left(10^{-3},\ q_{95}\left(\sqrt{u^2+v^2}\;|\;M=1\right)\right),
\]

where \(q_{95}\) is the 95th percentile over all valid input frames and cells. This is robust to wake deficits and avoids using the absolute maximum, which can be corrupted by PIV spikes.

Then I normalize

\[
\hat u = u/s,\qquad \hat v = v/s.
\]

The network predicts dimensionless future velocity, and the final physical prediction is multiplied by the same \(s\).

### Static context channels

I add six static channels at the input:

1. mask \(M\),
2. time-mean \(\hat u\) over the input window,
3. time-mean \(\hat v\) over the input window,
4. input-window TKE map  
   \[
   k_{\text{in}} = 0.5(\operatorname{var}_t \hat u + \operatorname{var}_t \hat v),
   \]
5. normalized row coordinate,
6. normalized column coordinate.

The last two help the network know absolute wake/probe locations because the airfoil is fixed.

The 20 input frames are flattened into channels:

\[
20 \times 2 = 40
\]

dynamic channels. With the 6 static channels, the network input is

\[
46 \times 32 \times 64.
\]

## Network

The network is a compact U-Net with residual blocks, GroupNorm, SiLU activations, and a small global context branch.

### Encoder

| stage | resolution | channels | blocks |
|---|---:|---:|---|
| stem | 32×64 | 48 | 3×3 conv, GN, SiLU |
| E1 | 32×64 → 16×32 | 48 → 64 | 2 residual blocks, stride-2 down |
| E2 | 16×32 → 8×16 | 64 → 96 | 2 residual blocks, stride-2 down |
| E3 | 8×16 → 4×8 | 96 → 128 | 2 residual blocks, stride-2 down |
| bottleneck | 4×8 | 128 | 2 residual blocks |

Each residual block is:

\[
\text{Conv}_{3\times 3} \rightarrow \text{GroupNorm} \rightarrow \text{SiLU}
\rightarrow \text{Conv}_{3\times 3} \rightarrow \text{GroupNorm}
\rightarrow \text{add residual} \rightarrow \text{SiLU}.
\]

### Global regime/context branch

At the bottleneck, I apply global average pooling:

\[
z \in \mathbb{R}^{128}.
\]

Then:

\[
e = W_2\,\text{SiLU}(W_1 z),
\]

with embedding dimension 64.

During training, if Reynolds and angle-of-attack metadata are available, I add a small auxiliary regression head from \(e\) predicting \(\log Re\) and angle of attack. This only regularizes the embedding. At inference this metadata is not used.

The embedding \(e\) modulates two decoder feature maps through FiLM:

\[
f \leftarrow f(1+\gamma(e)) + \beta(e),
\]

at the 8×16 and 16×32 decoder resolutions. This gives the network a lightweight way to adapt to regime information inferred from the input window.

### Decoder

Mirror of encoder:

| stage | resolution | channels |
|---|---:|---:|
| D3 | 4×8 → 8×16 | concat 128 + 96 → 96 |
| D2 | 8×16 → 16×32 | concat 96 + 64 → 64 |
| D1 | 16×32 → 32×64 | concat 64 + 48 → 48 |

Each decoder stage has nearest-neighbor upsample, 3×3 conv, then two residual blocks.

### Output head

Final 3×3 conv:

\[
48 \rightarrow 80
\]

channels.

These 80 channels are split into:

- 40 channels: dimensionless velocity correction,
- 40 channels: raw positive uncertainty scale.

The 40 velocity channels are reshaped to

\[
20 \times 32 \times 64 \times 2.
\]

## How the 20 output frames are produced

The network does not roll out the future one frame at a time. It emits all 20 frames simultaneously.

I use a residual output:

\[
\hat y_{\text{norm}} = \hat y_{\text{base}} + \Delta,
\]

where \(\hat y_{\text{base}}\) is the last normalized input frame repeated 20 times, and \(\Delta\) is the network output. The final physical prediction is

\[
y_{\text{phys}} = s\,\hat y_{\text{norm}}.
\]

Pressure is set to zero:

\[
p_{\text{pred}} = 0.
\]

Invalid cells are forced to zero using the mask.

The uncertainty head produces

\[
\sigma_{\text{norm}} = \text{softplus}(\text{raw}) + 10^{-3}.
\]

The physical standard deviation is

\[
\sigma_{\text{phys}} = s\,\sigma_{\text{norm}}.
\]

This is used for bounds, described below.

## Parameter count and inference cost

The model has approximately **1.8M–2.2M parameters**, depending on exact FiLM/head implementation.

FP32 weights are under 10 MB. TorchScript package is well below 256 MB.

On an RTX 2060, batch size 1, 32×64 input, FP16 inference, I expect roughly **5–15 ms per sample**, including scale/mask preprocessing. This is a guess, but the network is small and one-shot.

## Why this architecture over obvious alternatives

### Why not autoregressive ConvLSTM / recurrent rollout?

A recurrent model would require 20 sequential steps, increasing latency and accumulating errors. The metric penalizes the full 20-frame window, and the speed score penalizes wall time. A one-shot model is much more attractive under the 5-minute wall clock and the time-score formula.

### Why not a transformer?

The grid is tiny: 32×64. A transformer could work, but it is more likely to overfit on only ~170 trajectories, is less naturally masked, and is usually less sample-efficient for this kind of local spatial PDE-like task.

### Why not an FNO?

Fourier operators are appealing for fluids, but the domain is partially masked, the airfoil introduces strong non-periodic geometry, the real data is noisy, and I need a very small, fast, robust model. A U-Net with explicit mask channels is simpler and easier to train on a 6 GB laptop GPU.

### Why direct 20-to-20 instead of predicting dynamics?

The horizon is only 0.4 s, but it can contain substantial advection. Direct multistep prediction avoids rollout error and lets the loss directly target the scored window.

---

# 2. Training recipe

## Window construction

For each trajectory, form input/target pairs:

\[
\text{input}: [t-20, t-1],
\qquad
\text{target}: [t, t+19].
\]

Discard windows that cross trajectory boundaries.

### Sim split

- 100 trajectories, 1000 frames.
- Use stride 5.
- Approximately  
  \[
  100 \times \frac{1000-40}{5} \approx 19{,}200
  \]
  windows.

### Real split

- 81 trajectories, variable length.
- Use stride 3.
- Approximately 20k–25k windows depending on trajectory lengths.

I hold out real trajectories by Reynolds number, not randomly by window. If there are 18 real training Reynolds numbers, I keep 3 Reynolds numbers for validation and train on the remaining 15. The validation Re set should be treated as proxy for the scored unseen-Re condition.

## Use of simulated and real splits

I use a two-stage schedule:

1. **Pretrain on simulated data.**  
   Learn generic wake dynamics over a broad Reynolds and angle-of-attack grid.

2. **Fine-tune on real data.**  
   Adapt to PIV noise, real mask, real velocity scale, and real distribution.

During fine-tuning I mix real and simulated data with high real weight, for example a batch sampler with probability 0.75 real and 0.25 simulated. This prevents catastrophic forgetting of simulated regimes while prioritizing the scored domain.

## Dimensionalisation of simulated data

For simulated trajectory with Reynolds number \(Re\),

\[
U_\infty = \frac{Re \cdot 10^{-6}}{0.072}.
\]

I multiply simulated \(u,v\) by \(U_\infty\). Pressure is ignored.

If this produces a simulated speed distribution that is badly inconsistent with real data, I instead use per-trajectory robust scale matching:

\[
s_{\text{sim}} = q_{95}(\text{sim speed}),
\qquad
s_{\text{real}} = q_{95}(\text{real speed}),
\]

and rescale simulated trajectories so that their robust scale distribution matches the real one. The model itself always uses per-sample normalization, so this mostly matters for pretraining stability.

## Loss

Let \(y\) be the target normalized velocity field and \(\hat y\) the predicted normalized velocity field. All losses are masked to valid cells.

### Main velocity loss

\[
\mathcal{L}_{\text{vel}} =
\frac{1}{|M|}
\sum_{M}
\|\hat y - y\|_2^2.
\]

This is the primary loss.

### TKE-map loss

Predicted and target TKE maps are

\[
k_{\text{pred}} = 0.5(\operatorname{var}_t \hat u + \operatorname{var}_t \hat v),
\]

\[
k_{\text{tgt}} = 0.5(\operatorname{var}_t u + \operatorname{var}_t v).
\]

I use a relative-L2-style surrogate:

\[
\mathcal{L}_{\text{tke}}
=
\frac{\|k_{\text{pred}} - k_{\text{tgt}}\|_2}
{\|k_{\text{tgt}}\|_2 + \epsilon},
\]

averaged over the batch.

### Wake-mean / MVPE-proxy loss

If the 36 probe coordinates are available:

\[
\mathcal{L}_{\text{mvpe}}
=
\frac{
\|\overline{\hat u}_{P} - \overline{u}_{P}\|_2^2
+
\|\overline{\hat v}_{P} - \overline{v}_{P}\|_2^2
}
{
\|\overline{u}_{P}\|_2^2
+
\|\overline{v}_{P}\|_2^2
+
\epsilon
}.
\]

If probe coordinates are unavailable, I approximate \(P\) with a downstream wake ROI mask: valid cells downstream of the airfoil, centered vertically on the wake. This is a declared approximation.

### Auxiliary regime regression

If training metadata is available:

\[
\mathcal{L}_{\text{regime}}
=
\text{MSE}(\widehat{\log Re}, \log Re)
+
\text{MSE}(\hat\alpha, \alpha).
\]

This is only a regularizer. It is not used at inference.

### Uncertainty loss

After the deterministic model is stable, I add the uncertainty head and train with masked Gaussian negative log-likelihood:

\[
\mathcal{L}_{\text{nll}}
=
\frac{1}{|M|}
\sum_M
\left[
\frac{(y-\mu)^2}{2\sigma^2}
+
\log \sigma
\right].
\]

This head is calibrated separately for SPS.

### Total loss

Deterministic stage:

\[
\mathcal{L}
=
\mathcal{L}_{\text{vel}}
+
0.5\,\mathcal{L}_{\text{tke}}
+
0.5\,\mathcal{L}_{\text{mvpe}}
+
0.05\,\mathcal{L}_{\text{regime}}.
\]

Uncertainty stage:

\[
\mathcal{L}
=
\mathcal{L}_{\text{vel}}
+
0.5\,\mathcal{L}_{\text{tke}}
+
0.5\,\mathcal{L}_{\text{mvpe}}
+
0.2\,\mathcal{L}_{\text{nll}}.
\]

## Optimizer and schedule

All training uses PyTorch-only operations. No scipy, sklearn, pandas, einops, or OpenCV are required.

### Stage 1: simulated pretraining

- optimizer: AdamW,
- learning rate: \(2\times 10^{-4}\),
- weight decay: \(5\times 10^{-4}\),
- batch size: 16,
- mixed precision: yes,
- gradient clipping: 1.0,
- steps: about 12k,
- cosine decay to \(10^{-5}\).

This should be feasible in about one to two hours on a degraded RTX 2060 at ~70% sustained utilization.

### Stage 2: real fine-tuning

- optimizer: AdamW,
- learning rate: \(5\times 10^{-5}\),
- batch size: 16,
- real:sim sampling ratio: 3:1,
- steps: about 8k–10k,
- cosine decay to \(10^{-6}\),
- early stop on leave-Re-out real validation rel_l2.

### Stage 3: uncertainty calibration

- initialize from best deterministic checkpoint,
- train uncertainty head for 2k–3k steps,
- learning rate: \(1\times 10^{-5}\),
- backbone learning rate optionally reduced by a factor of 5,
- then calibrate interval width on validation.

## Augmentation

Augmentation is allowed. I use only augmentations that respect the fixed geometry and scoring.

### Input measurement noise

Estimate PIV noise from real data, see measurement #1 below. Then:

- for simulated inputs: add Gaussian noise with standard deviation sampled from  
  \[
  [0, 1.5]\sigma_{\text{noise}},
  \]
- for real inputs: add Gaussian noise with standard deviation sampled from  
  \[
  [0, 0.5]\sigma_{\text{noise}},
  \]
  to avoid double-counting measurement noise already present in the target.

Noise is added to inputs only, not to targets. The target is the measured future, and the model should predict the conditional expectation of that measured future.

### Missing-patch dropout

Randomly zero 0–2% of valid input cells per sample and update the mask channel. This improves robustness to small PIV holes or spurious zero patches.

### Mild spatial smoothing of simulated inputs

With probability 0.3, apply a small Gaussian blur to simulated input velocity fields, with kernel width about 0.5–1 pixel. This mimics PIV spatial smoothing. Implemented with a fixed PyTorch convolution kernel.

### What I deliberately do not do

- No random translations or rotations. The airfoil and probe locations are fixed.
- No horizontal flips. The airfoil geometry and wake direction are not symmetric.
- No vertical flips unless the data shows signed angle-of-attack symmetry; the stated angles are 0–20 degrees, so I assume no negative-angle counterpart.
- No time reversal. The flow is dissipative and not time-reversible.

---

# 3. Interval-bound strategy for SPS

I supply bounds on every call. The default ±5% of \(|\text{prediction}|\) is too crude near zero velocity and too wide or too narrow elsewhere.

## Predictive distribution

The model outputs a mean \(\mu_i\) and standard deviation \(\sigma_i\) for each scored element \(i\), where \(i\) indexes future time, spatial location, and velocity component.

The raw interval half-width is

\[
h_i = k\,\sigma_i,
\]

where \(k\) is calibrated on validation data.

The bounds are

\[
\ell_i = \mu_i - h_i,
\qquad
u_i = \mu_i + h_i.
\]

I clamp half-widths:

\[
h_i = \operatorname{clamp}(h_i,\ h_{\min},\ h_{\max}).
\]

Initial values before validation tuning:

\[
h_{\min} = 0.02\,\sigma_{\text{global}},
\]

\[
h_{\max} = 0.6\,\sigma_{\text{global}},
\]

with

\[
\sigma_{\text{global}} = 0.0563870259.
\]

So numerically:

\[
h_{\min} \approx 0.00113\ \mathrm{m/s},
\]

\[
h_{\max} \approx 0.0338\ \mathrm{m/s}.
\]

The total interval width is

\[
w_i = u_i - \ell_i = 2h_i.
\]

The SPS nil is

\[
\text{nil}_i = \frac{w_i}{\sigma_{\text{global}}}.
\]

## Reasoning about width

If the forecast error at element \(i\) is approximately Gaussian with standard deviation \(\sigma_i\), and the interval is symmetric around the mean, the coverage probability is

\[
P_{\text{cover}}(h_i)
=
\operatorname{erf}
\left(
\frac{h_i}{\sqrt{2}\sigma_i}
\right).
\]

Ignoring the accuracy-dependent multiplier \((1-pm)\), which does not depend on interval width for a fixed point prediction, the expected SPS contribution is approximately

\[
\mathbb{E}[S_i]
\propto
\operatorname{erf}
\left(
\frac{h_i}{\sqrt{2}\sigma_i}
\right)
\exp\left(
-\frac{2h_i}{\sigma_{\text{global}}}
\right).
\]

This expression shows the tradeoff:

- if \(h_i\) is too small, coverage collapses and the element scores zero;
- if \(h_i\) is too large, the exponential width penalty destroys the score.

For small predictive uncertainty \(\sigma_i \ll \sigma_{\text{global}}\), the optimal interval is only a few predictive standard deviations wide, not a fixed fraction of the prediction magnitude. For large \(\sigma_i\), the interval should not be allowed to grow without bound, because once \(w_i\) is much larger than \(\sigma_{\text{global}}\), the exponential term becomes tiny.

Therefore I use a heteroscedastic interval with a floor and a soft cap.

## Calibration procedure

On the real leave-Re-out validation set:

1. Compute standardized residuals  
   \[
   z_i = \frac{y_i-\mu_i}{\sigma_i}.
   \]

2. If the standardized distribution is overconfident, apply temperature scaling:  
   \[
   \sigma_i \leftarrow T\sigma_i.
   \]  
   A simple choice is to match the empirical median absolute standardized residual to the Gaussian expectation:
   \[
   T = \frac{\operatorname{median}(|z|)}{0.6745}.
   \]

3. Grid-search \(k\), \(h_{\min}\), and \(h_{\max}\) to maximize empirical SPS on validation.

Default search ranges:

\[
k \in [0.5, 2.5],
\]

\[
h_{\min} \in [0,\ 0.05\sigma_{\text{global}}],
\]

\[
h_{\max} \in [0.3\sigma_{\text{global}},\ 1.5\sigma_{\text{global}}].
\]

Because SPS depends on actual coverage, I prefer validating on real data rather than relying purely on the Gaussian formula.

## Behavior near zero velocity

The default ±5% of \(|\text{prediction}|\) becomes dangerously narrow when the predicted velocity is near zero. This matters in the wake and near stagnation regions. The floor \(h_{\min}\) is specifically there to keep nonzero coverage in those cells.

## Pressure bounds

I output

\[
p_{\text{pred}} = 0.
\]

For pressure bounds I use a negligible interval around zero, for example

\[
[-10^{-6}, 10^{-6}].
\]

I assume pressure is not materially scored. If it is, this is harmless but I do not design the system around exploiting it.

---

# 4. First three measurements before writing model code

These are cheap data diagnostics. Each one can change the design.

---

## Measurement 1: velocity scale, Reynolds consistency, and PIV noise floor

### What I compute

For every real trajectory:

1. Valid mask from zeros.
2. Robust speed scale \(q_{95}\).
3. Upstream mean velocity vector, if identifiable.
4. Temporal increments  
   \[
   \Delta u_t = u_{t+1} - u_t,
   \qquad
   \Delta v_t = v_{t+1} - v_t.
   \]
5. High-frequency noise estimate from upstream regions or from second differences:
   \[
   \sigma_{\text{noise}}
   \approx
   \operatorname{median}
   \left(
   \frac{|u_{t+1}-2u_t+u_{t-1}|}{\sqrt{6}}
   \right)
   \]
   in regions where the flow should be smooth.

For simulated data, I compare the nominal \(U_\infty = Re\nu/c\) scaling against the real speed distribution.

### Design consequences

If real speed scales match \(Re\nu/c\), I keep that dimensionalisation.

If they do not, I switch to empirical per-trajectory scaling and match simulated robust speed distributions to real ones.

If \(\sigma_{\text{noise}}\) is large, for example above \(0.02\ \mathrm{m/s}\), I increase input noise augmentation and raise \(h_{\min}\) for SPS.

If \(\sigma_{\text{noise}}\) is very small, I lower \(h_{\min}\) to improve SPS.

---

## Measurement 2: predictability baseline and advection timescale

### What I compute

On real training trajectories, holding out Reynolds numbers where possible:

1. Persistence baseline: repeat the last input frame 20 times.
2. Linear extrapolation baseline:
   \[
   \hat y_{t+k} = y_t + k(y_t-y_{t-1}).
   \]
3. Optional frozen-advection baseline: estimate a bulk displacement from recent frames and shift the last frame downstream.
4. Temporal autocorrelation of velocity fields.
5. Dominant wake frequency from temporal FFT of wake-region averages.

I compute rel_l2, TKE error, and wake-mean error for these cheap baselines.

### Design consequences

If persistence already gives a surprisingly low rel_l2, the residual direct U-Net is likely sufficient and I keep the simple repeated-last-frame base.

If linear or frozen-advection baseline is much better than persistence, I modify the output base from repeated last frame to an advected or linearly extrapolated base. This is a cheap change: the network still predicts a correction, but the correction is around a better prior.

If the flow decorrelates much faster than 0.4 s, I widen uncertainty intervals and put more weight on TKE and wake-mean losses, because exact phase matching becomes harder.

---

## Measurement 3: domain gap, mask geometry, and regime identifiability

### What I compute

1. The real zero mask and its intersection with the simulated grid.
2. Whether simulated fields contain the same airfoil/FOV mask.
3. Spatial mean, variance, and TKE maps for sim versus real.
4. Simple regime identifiability: using only input-window statistics such as mean speed, TKE, wake variance, and dominant frequency, can I predict Reynolds number and angle of attack with a nearest-centroid or linear probe?
5. Downstream wake variability, to choose the MVPE-proxy ROI if exact probe coordinates are unavailable.

### Design consequences

If sim and real masks differ, I rely more heavily on the explicit mask channel and real fine-tuning. I also ensure all losses are mask-weighted.

If regime is strongly identifiable from the input window, I keep the auxiliary regime regression and FiLM conditioning.

If regime is weakly identifiable, I remove the auxiliary regime loss and widen SPS bounds, because the model cannot reliably infer the regime from the window.

If the wake ROI has very different mean behavior than the full field, I increase the weight of the wake-mean/MVPE-proxy loss.

---

# 5. Predicted subscores

These are honest guesses. I have not seen the data. The ranges reflect uncertainty in flow predictability, noise, and unseen-Re generalization.

## Expected error assumptions

I assume the final model achieves approximately:

- relative velocity L2 error: 0.30–0.45,
- relative TKE-map error: 0.45–0.70,
- relative MVPE error: 0.25–0.40,
- inference time: 8–20 ms per sample,
- SPS: moderate calibration success.

Using

\[
\text{score} = \frac{100}{1+0.5e},
\]

and

\[
\text{time\_score}
=
\frac{100}{1+\sqrt{t/0.72896}},
\]

I get the following rough predictions.

| subscore | central estimate | plausible uncertainty range | main driver |
|---|---:|---:|---|
| rel_l2 | 82–86 | 75–90 | wake phase error and sim-to-real gap |
| tke | 70–76 | 62–82 | deterministic blurring of unsteady wake |
| mvpe | 80–85 | 72–89 | time-mean is easier than instantaneous fields |
| time_score | 85–89 | 78–92 | depends on preprocessing and CUDA overhead |
| SPS | 45–55 | 30–65 | calibration quality and coverage-width tradeoff |

I would not trust a point estimate tighter than that.

## Which subscore to attack first

I would attack **SPS** first.

Reason:

- It is currently the weakest expected subscore.
- It is directly optimizable using validation data.
- Improving bounds does not necessarily hurt accuracy or speed.
- The formula is known, so grid-searching interval parameters is cheap.
- A default ±5% bound is likely poor near zero velocity and in the wake, which are important regions here.

If the unknown final-score aggregation turns out to weight accuracy much more strongly than calibration, then the next best attack would be **TKE**, because deterministic regression tends to underpredict temporal variance. But given only the information in the prompt, SPS has the best expected marginal gain per unit effort.

---

# 6. Two biggest risks and cheapest falsifying experiments

## Risk 1: unseen-Reynolds and sim-to-real generalization failure

The scored set is real data at Reynolds numbers not present in training. The simulated split has cleaner fields, pressure available, and possibly different small-scale dynamics. The model may learn Reynolds-specific features that do not transfer.

### Consequence

Large rel_l2 and MVPE degradation on unseen Reynolds numbers, and poor SPS coverage because uncertainty is underestimated.

### Cheapest falsifying experiment

Train a reduced version of the model, or train the full model for fewer steps, with a leave-Reynolds-out split of the real training data.

Measure:

\[
\frac{\text{rel\_l2}_{\text{unseen Re}}}
{\text{rel\_l2}_{\text{seen Re}}}.
\]

If the unseen-Re error is more than about 15–20% worse than seen-Re error, the design is at risk.

### Mitigation if falsified

- increase real data weighting,
- add stronger input-noise augmentation,
- widen SPS bounds for uncertain regimes,
- use more explicit regime embedding,
- reduce reliance on simulated pressure or simulated-only regimes,
- possibly train with Reynolds-dropout so the model learns regime-invariant features.

---

## Risk 2: deterministic direct forecast underpredicts wake unsteadiness

A direct MSE-trained model tends to predict the conditional mean. For a chaotic or partially unpredictable wake, this can blur vortex shedding and reduce temporal variance. That hurts TKE specifically, and can also cause interval misses.

### Consequence

TKE subscore is much lower than rel_l2 subscore. Predicted TKE map may be systematically below target TKE. SPS coverage may fail in high-variance wake regions.

### Cheapest falsifying experiment

After a short training run, compute on validation:

\[
R_{\text{TKE}}
=
\frac{
\sum k_{\text{pred}}
}{
\sum k_{\text{tgt}}
}.
\]

Also compute empirical coverage of candidate intervals.

Red flags:

\[
R_{\text{TKE}} < 0.7
\]

or coverage below about 70% with \(k\approx 1\).

### Mitigation if falsified

- increase TKE loss weight,
- add a spectral or temporal-variance preserving loss,
- use quantile regression instead of Gaussian NLL,
- use a small ensemble distilled into the uncertainty head,
- widen calibrated bounds,
- if severe, switch the output base from repeated last frame to a short autoregressive refinement for the last few frames, although this sacrifices speed.

---

# Final committed design in one paragraph

Use a compact one-shot 20-to-20 scale-conditioned residual U-Net with about 2M parameters. Normalize each sample by its own robust velocity scale, ignore pressure, predict dimensionless velocity corrections relative to the last input frame, rescale to physical units, and output zero pressure. Pretrain on dimensionalised simulated trajectories, fine-tune on real PIV trajectories with noise augmentation and a leave-Reynolds-out validation discipline. Train with velocity MSE plus explicit TKE-map and wake-mean losses. Add a heteroscedastic uncertainty head, calibrate interval widths on real validation data to maximize SPS, and supply bounds on every call. The first diagnostics should measure velocity scale/noise, baseline predictability/advection, and sim-real mask/regime gap. Expected subscores are roughly high 70s to high 80s for accuracy, mid-80s for time if inference stays near 10 ms, and 40–60 for SPS, with SPS being the first subscore I would attack.