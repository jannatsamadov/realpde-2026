
https://github.com/AI4Science-WestlakeU/RealPDEBench

https://www.codabench.org/competitions/17385/
https://www.codabench.org/competitions/17363/#/pages-tab
https://huggingface.co/datasets/AI4Science-WestlakeU/RealPDE-Competition-Data






NeurIPS 2026 RealPDE Competition - Track 1: Sim2Real
Part of the NeurIPS 2026 RealPDE Competition. See the competition website for both tracks, timeline, and announcements. Track 2 (LTTTA) runs on Codabench: codabench.org/competitions/17385.

Registration required. Registration via the team form is required for both tracks (deadline: August 20, 2026):

Track 1 registration form
Track 2 registration form
Joining on Codabench alone is not registration: after the registration deadline, Codabench join requests without a submitted form will be denied.

Forecasting complex physical systems is a central challenge in science and engineering. Modern Scientific ML models are often trained and evaluated primarily on simulated data, but real deployments face noisy measurements, partial observability, and distribution shifts between numerical simulation and real-world experiments.

Track 1 focuses on Simulation-to-Real transfer learning for NACA4418 airfoil flow. Participants develop models that use simulated CFD data and real PIV measurements to predict real-world fluid dynamics accurately, efficiently, and robustly.

Announcements
This section is updated throughout the competition. Check back regularly; substantive changes are also announced on the competition forum.

Development Phase restarted (5 August).

A new Track 1 Development Phase opens on 5th August 2026, 00:00 UTC with an empty leaderboard. The scores you have now were computed on the old evaluation set with the old scoring rule, so please submit again. The earlier phase and its leaderboard stay visible and frozen. The restart moves no other date: registration still closes on 20th August 2026, and the Development Phase still closes at 24:00 UTC on 27th September 2026.

We rebuilt the evaluation set because the windows were cut at a stride shorter than the prediction horizon, which put one window's target inside another window's input. A model reading across the windows it was handed could find an answer it had already been given. The fault is ours, in how we cut the windows. Teams who spent the phase building models deserve a leaderboard that reflects their work.

Multiple teams flagged the overlap during the phase and shared what they found. We are grateful to them for debugging the RealPDE Competition for us. A scientific benchmark stays honest because its community keeps checking it. Please keep telling us when something looks wrong, and we will keep fixing it.

What to read before you resubmit:

[Submission] predict may now be called more than once, each call in a fresh isolated subprocess, and metadata is an empty dictionary on scored calls. Code that indexes into metadata needs a change.
[Evaluation] sps_score now maps its aggregate to 0-100 linearly, replacing the logistic it used before, so every SPS score on the old board moves. Targets outside the PIV field of view or inside the airfoil body are not scored in SPS.
How the total is formed. The leaderboard shows a single final_score, and your own submission's detailed results show the five subscores it is combined from. The combination is not published. The starting kit computes the five subscores exactly as the leaderboard does, so you can still check those on your own data.
[Rules] The execution time limit is now 5 minutes, raised from 3. Training data must come from the competition's own release: no self-generated data, and no pretrained models trained on anything else. Checkpoints trained from the release are permitted, including the official baselines. Standard augmentation of the release is allowed; see the FAQ.
Starting kit v9 replaces the copy you have. Download it from this phase.
2026-08-09: Data augmentation clarified. Standard augmentation of the released data is allowed in both tracks. Every training sample must trace back to the released training data. The full boundary is in a new FAQ entry, linked from the Rules page.

2026-08-05: Evaluation set rebuilt, scoring updated, Development Phase restarted. Starting kit v9. See the notice above.

2026-07-22: Starting kit v6. Adds a worked example of returning optional SPS interval bounds (lower / upper) from predict(). Interface and scoring are unchanged.

2026-07-18: Warm-up submissions cleared. Due to an operational issue on the organizer side, the submissions in the Warm-up Phase were cleared during a maintenance action. This affects the non-scoring Warm-up Phase only. If you had submissions in the Warm-up Phase, you are welcome to re-submit at any time. Apologies for the inconvenience.

2026-07-05: Competition launched. Warm-up Phase is open.

At A Glance
Dataset: paired real-world and simulated trajectories of 3D NACA4418 airfoil cross-sectional flow.
Task: predict real-world future flow fields from input flow windows.
Modalities: simulation provides u, v, and p; real measurements provide u and v, with p treated as unmeasured.
Evaluation resolution: approximately 32 x 64 after spatial subsampling.
Ranking: final_score is a single 0-100 score combined from five 0-100 subscores (Rel-L2, TKE, MVPE, Time, SPS). The combination is not published.
Schedule
Warm-up Phase: July 5 - July 19, 2026 (UTC). Format validation and platform smoke test on public RealPDEBench foil data.
Main Development Phase: July 20 - September 27, 2026 (UTC). Hidden validation leaderboard.
Final Decision Phase: September 28 - October 25, 2026 (UTC). The top 10 teams from the development leaderboard are shortlisted; organizers re-train their methods and evaluate on a private test set to decide the top 3.
Final results announced: November 10, 2026.
Code submission deadline: November 25, 2026.
NeurIPS presentation: December 6, 2026.
Prizes
The RealPDE prize pool awards 6,000 / 3,000 / 1,500 USD to the top three teams of each track (funded by Uniforce AI Ltd.). The top 3 teams in each track are invited to deliver oral presentations at the competition workshop. The top 5 teams in each track are invited to co-author a joint results paper submitted to a relevant academic journal or the NeurIPS Datasets and Benchmarks Track, and each of these teams may nominate one advisor of their own (for example, a faculty supervisor) as a co-author. Certificates are awarded to the top 5 teams in each track and may optionally name one advisor.

Resources
Start from the downloadable Track 1 starting kit available on the Files tab of this competition. The Files tab is only visible after you sign in and your participation request is approved.

It vendors the baseline models and the official scoring program, so it builds, scores, and packages a submission offline:

starting_kit/
├── README.md               data format, geometry, and usage notes
├── submission_template.py  Track 1 predict() API; rename to submission.py
├── load_baseline.py        load CNO / FNO / Transolver checkpoints and run a forward pass
├── pack_ckpt_fp16.py       complex-safe fp16 packing to fit the 256 MB size cap
├── scoring.py              computes the five leaderboard subscores locally
├── smoke_test_kit.py       self-check: clean-room import + baseline load + forward
├── _vendor/einops/         vendored einops (the Transolver forward needs it; offline)
└── rpde_baselines/         vendored CNO / FNO / Transolver model code (imports offline)
Key entry points:

Build: implement predict(input_array, metadata) in submission_template.py (it ships a persistence baseline), or wrap a pretrained baseline via load_baseline.py. Rename to submission.py before zipping.
Score locally: scoring.py computes the five subscores exactly as the leaderboard does; run it on your predictions.npz plus a reference targets.npz (see the kit README).
Fit the size cap: pack_ckpt_fp16.py packs an fp32 checkpoint (only FNO needs it: 403 MB to 201 MB) under the 256 MB limit.
Pretrained baseline checkpoints (CNO / FNO / Transolver, sim-pretrained and real-finetuned) ship with the training release on Hugging Face, mirrored on Google Drive. All were trained with the RealPDEBench code. Track 1 and Track 2 share the same data release.

Contact
realpde-competition@googlegroups.com, or the competition forum.



Benchmarks/Competitions
Datasets
Competition Logo
NeurIPS 2026 RealPDE Competition - Track 2: Long-Term Test-Time Adaptation

$21,000 USD prize pool across RealPDE tracks
Organized by: tailin_wu (realpde-competition@googlegroups.com)
Current Phase Ends: September 28, 2026 at 3:59 AM GMT+4
Current server time: August 13, 2026 at 9:22 PM GMT+4
Docker image: w3nhao/realpde-track2@sha256:f07d1d68e19cb7ed5f2f405b6775f4fafb985887ac78356a9599f35365ccf972 
NeurIPS 2026 RealPDE Competition - Track 2: LTTTA
Part of the NeurIPS 2026 RealPDE Competition. See the competition website for both tracks, timeline, and announcements. Track 1 (Sim2Real) is already live on Codabench: codabench.org/competitions/17363.

Registration required. Registration via the team form is required for both tracks (deadline: August 20, 2026):

Track 1 registration form
Track 2 registration form
Joining on Codabench alone is not registration: after the registration deadline, Codabench join requests without a submitted form will be denied.

Track 2 - Long-Term Test-Time Adaptation (LTTTA) studies long-horizon autoregressive prediction with continuous adaptation to streaming real-world observations. Your model walks through real PIV trajectories window by window: at each step it predicts the next 20 frames, and after each prediction the ground truth of the previous window is revealed for adaptation. Agent-augmented methods (bounded controllers deciding when and how to adapt, optionally through an organizer-provided LLM gateway) are explicitly welcome.

Announcements
This section is updated throughout the competition. Check back regularly; substantive changes are also announced on the competition forum.

Development Phase restarted (5 August).

A new Track 2 Development Phase opens on 5th August 2026, 00:00 UTC with an empty leaderboard. Scoring has been updated, so please submit again. The earlier phase and its leaderboard stay visible and frozen. The restart moves no other date: registration still closes on 20th August 2026, and the Development Phase still closes at 24:00 UTC on 27th September 2026.

Multiple teams flagged problems on Track 1 during the phase and shared what they found. We are grateful to them for debugging the RealPDE Competition for us. A scientific benchmark stays honest because its community keeps checking it. Please keep telling us when something looks wrong, and we will keep fixing it.

Your submission still runs unmodified. The ttt_step and reset_ttt_state signatures, the returned dictionary, the evaluation data and the 10-minute execution limit are all as they were. What changed is how the run is scored:

[Evaluation] sps_score now maps its aggregate to 0-100 linearly, replacing the logistic it used before, so every SPS score on the old board moves. Targets outside the PIV field of view or inside the airfoil body are not scored in SPS. reset_ttt_state is now timed, charged to the step that follows the trajectory boundary.
How the total is formed. The leaderboard shows a single final_score, and your own submission's detailed results show the five subscores it is combined from. The combination is not published. The starting kit computes the five subscores exactly as the leaderboard does, so you can still check those on your own data.
[Rules] Training data must come from the competition's own release: no self-generated data, and no pretrained models trained on anything else. Checkpoints trained from the release are permitted, including the official baselines. Standard augmentation of the release is allowed; see the FAQ.
Starting kit v6 replaces the copy you have. Download it from this phase; older copies of scoring.py and local_eval.py are out of date.
2026-08-12: Data issue in train_real/7575_0.h5. That file reuses the data from 6300_0, so it does not hold measurements of case 7575. Please exclude it from training. Only this file is affected: the other angles of attack at Re 7575 are fine, and so is 6300_0 itself. Case 7575 at 0 degrees cannot be measured again at present, so the file stays in the release as published and is simply not to be used. Thanks to the team who inspected the data and told us.

2026-08-09: Data augmentation clarified. Standard augmentation of the released data is allowed in both tracks. Every training sample must trace back to the released training data. The full boundary is in a new FAQ entry, linked from the Rules page.

2026-08-05: Scoring updated, Development Phase restarted. Starting kit v6. See the notice above.

2026-07-22: Starting kit v4 ships a worked example, updated interface docs, and local-eval support for bounds.

2026-07-22: SPS interval bounds now supported in Track 2. ttt_step can now return optional per-element lower / upper bounds via the info dict, in the same normalized space and shape as pred_norm (both keys together, on every step). The SPS formula, its weights, and the default ±5%·|pred| band for point-only submissions are unchanged, and existing submissions are unaffected.

2026-07-05: LLM gateway online. Agents may call an organizer-provided OpenAI-compatible LLM gateway. Usage and a code example are on the Submission page; restrictions are in the Rules.

2026-07-05: Competition launched. Warm-up Phase is open.

Task
Data: real-world PIV flow around a NACA4418 airfoil.
Streaming protocol: strict trajectory order, batch size 1, windows of 20 input frames to 20 output frames.
At step t your model receives the current input window (normalized) plus the ground truth of step t-1, and must return the prediction for step t.
Adaptation (gradient steps, calibration, memory, expert selection, LLM-guided decisions) happens inside your ttt_step; all of it counts toward the Time metric.
Ranking: final_score is a single 0-100 score combined from five 0-100 subscores (Rel-L2, TKE, MVPE, Time, SPS). The combination is not published.
Phases
Warm-up Phase: July 5 - July 19, 2026 (UTC). Format validation and smoke test.
Main Development Phase: July 20 - September 27, 2026 (UTC). Hidden validation leaderboard.
Final Decision Phase: September 28 - October 25, 2026 (UTC). The top 10 teams from the development leaderboard are shortlisted; organizers re-train their methods and evaluate on a private test set to decide the top 3.
Final results announced: November 10, 2026.
Code submission deadline: November 25, 2026.
NeurIPS presentation: December 6, 2026.
All phase boundaries are in UTC (see FAQ).

Prizes
Track 2 shares the RealPDE prize pool: 6,000 / 3,000 / 1,500 USD for the top three teams of each track (funded by Uniforce AI Ltd.). The top 3 teams in each track are invited to deliver oral presentations at the competition workshop. The top 5 teams in each track are invited to co-author a joint results paper submitted to a relevant academic journal or the NeurIPS Datasets and Benchmarks Track, and each of these teams may nominate one advisor of their own (for example, a faculty supervisor) as a co-author. Certificates are awarded to the top 5 teams in each track and may optionally name one advisor.

Resources
Start from the downloadable starting kit available on the Files tab of this competition. The Files tab is only visible after you sign in and your participation request is approved.

It contains everything needed to build, locally score, and submit a Track 2 entry:

starting_kit/
├── README.md               quickstart and file guide
├── submission_template.py  copy to submission.py; correct prev-target TTT reference
├── ttt_model.py            optional TTT base class (interface + shape/timing contract)
├── local_eval.py           CPU smoke test on example_data; calls scoring.py for real subscores
├── scoring.py              computes the five leaderboard subscores locally
├── load_baseline.py        build CNO / FNO / Transolver and load an official checkpoint
├── pack_ckpt_fp16.py       complex-safe fp16 packing to fit the 256 MB size cap
├── docs/
│   ├── interface.md        full submission interface contract
│   └── metrics.md          metric summary (the Evaluation page is authoritative)
├── agentic_demo/           submittable agentic baseline: bounded controller + optional gateway LLM
│   ├── submission.py
│   ├── policy.yaml
│   └── README.md
├── rpde_baselines/         vendored CNO / FNO / Transolver model code (imports offline)
└── example_data/           two tiny synthetic trajectories + official normalization stats
    ├── mean_std_real.pt
    ├── make_example.py
    └── test_real/*.h5
Key entry points:

Build: copy submission_template.py to submission.py and wrap your model, or start from agentic_demo/ for a controller-based method. Use load_baseline.py to adapt a pretrained CNO / FNO / Transolver.
Score locally: python local_eval.py --submission . runs your model through the streaming loop and reports the five real subscores via the bundled scoring.py (illustrative on the tiny example data; point --data at real data for comparable numbers).
Fit the size cap: pack_ckpt_fp16.py packs an fp32 checkpoint to fp16 under the 256 MB limit.
Pretrained baseline checkpoints (CNO / FNO / Transolver, sim-pretrained and real-finetuned) ship with the training release on Hugging Face, mirrored on Google Drive. All were trained with the RealPDEBench code. Track 1 and Track 2 share the same data release.

Contact
realpde-competition@googlegroups.com, or the competition forum.

Chasuite
Competitions v1.6
Chahub
Chagrade
About
About
Github
Privacy and Terms
API Docs
CodaBench
Join us on Github for contact & bug reports

Questions about the platform? See our Docs for more information.

v1.31