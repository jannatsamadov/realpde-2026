Evaluation
Submissions are ranked by final_score, a 0-100 score where higher is better. It is combined from the five 0-100 subscores below, and the combination is not published.

The starting kit ships scoring.py, which computes those five subscores exactly as the leaderboard computes them, so you can check them locally before submitting.

Scores You See
The public leaderboard shows a single column: final_score. Your own submission's detailed results additionally show the five subscores it combines:

rel_l2_score: data fidelity score from relative L2 error.
tke_score: turbulent kinetic energy score.
mvpe_score: mean velocity profile error score.
time_score: runtime efficiency score.
sps_score: safe prediction score.
Metrics
Relative L2
Relative L2 measures prediction error on measured channels:

||pred - target||_2 / ||target||_2
For real airfoil data, only u and v are scored.

TKE
Turbulent kinetic energy error compares the kinetic energy fields induced by velocity fluctuations:

TKE = 0.5 * (mean_t((u - mean_t(u))^2) + mean_t((v - mean_t(v))^2))
The raw TKE error is reported as a relative L2 error.

MVPE
Mean velocity profile error compares time-averaged velocity profiles at probe locations behind the airfoil. Lower MVPE indicates better long-horizon physical consistency.

Time
Runtime is measured as mean neural inference wall time per evaluated sample. With r = t_neural / t_numerical and the reference numerical solver time t_numerical = 0.72896 seconds, the score is:

100 * 1 / (1 + sqrt(r))
SPS
The Safe Prediction Score (sps_score) rewards predictions that are accurate and paired with tight, well-calibrated intervals.

You may return per-element bounds lower / upper in predictions.npz (same shape as the measured prediction). If you do not, the scorer uses a default band lower = pred - 0.05*|pred|, upper = pred + 0.05*|pred| (width 0.1*|pred|). Bounds are all-or-nothing across the whole run: predict may be called more than once, and if some calls return bounds and others do not, every bound is discarded and the default band is used instead. Bounds whose shape does not match the prediction fail the run; bounds that are non-finite or reversed (lower > upper) score zero.

Let sigma_global = 0.0563870 (frozen for the season; the mean of the u, v channel standard deviations on the official train_real split). Per element, inside = (lower <= target <= upper) and nil = (upper - lower) / sigma_global. SPS sums three error branches (DM = Rel-L2, TKE, MVPE); each per-window error e is squashed by pm = e / (0.5 + e), and its elements are rewarded by (1 - pm) * exp(-nil) where inside, else 0:

branch    = mean over scored elements of (1 - pm) * exp(-nil), gated by inside
weighted  = 0.5*branch_dm + 0.3*branch_tke + 0.2*branch_mvpe
sps_score = 100 * weighted
Targets outside the PIV field of view or inside the airfoil body are not scored in SPS.

Because pm is bounded in [0, 1) and exp(-nil) is at most 1, every element contributes between 0 and 1, and the three branch weights sum to 1, so weighted lands in [0, 1] and sps_score in [0, 100]. An off-target or uncovered prediction contributes 0, never a negative value, so narrow intervals around bad predictions cannot buy score.

Execution Time Limit
Each submission runs under a wall-clock execution limit of 5 minutes in the Warm-up and Development phases, measured on container execution only (data download is excluded). A submission exceeding the limit is marked Failed and receives no score. This limit is independent of the Time metric above, which measures per-sample inference time inside the run.

Score Mapping
For Rel-L2, TKE, and MVPE, lower raw error maps to a higher 0-100 score:

score = 100 / (1 + 0.5 * error)
The five 0-100 subscores are then combined into a single 0-100 final_score. The combination is not published, so scoring.py stops at the five subscores.

Zero-Score Conditions
A submission scores 0 on every subscore if any of these hold, so guard against them:

the prediction contains non-finite values (NaN / Inf);
the prediction shape does not match the reference (N, 20, 32, 64, 3);
supplied lower / upper bounds are non-finite, or reversed (lower > upper).
The reason appears in your submission's detailed results.

A run fails outright and receives no score at all, rather than a zero, if bounds are supplied whose shape does not match the prediction, or if a prediction's sample count does not match the inputs it was given.