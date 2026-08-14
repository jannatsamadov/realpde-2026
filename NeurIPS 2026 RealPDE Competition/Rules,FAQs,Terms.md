Rules
Eligibility
The challenge is open to individuals and academic or industrial teams worldwide. Each team may have at most three members. Multiple accounts for the same team or individual are prohibited.

Anyone who has had access to the hidden evaluation data, to the scoring configuration, or to the private test set may not compete or share in a prize. This covers the organizers and anyone they have given that access to, whatever their affiliation. It is about access, not about where you work: a colleague of an organizer who has had none of it is eligible like anyone else.

Submission Limits
Warm-up Phase: each team is limited to 3 submissions per day.
Main Development Phase: each team is limited to 1 submission per day.
Each team is limited to 100 submissions per phase.
Submission archives must stay under 256 MB after extraction; oversized submissions are rejected without evaluation.
Each submission must finish evaluation within the execution time limit (5 minutes in the Warm-up and Development phases, container execution only; data download time is not counted). Submissions exceeding the limit are marked Failed and receive no score.
torch.utils.data.DataLoader must use num_workers=0.
Multiple accounts for the same team or individual are prohibited.
Evaluation Data Integrity
Your model is handed the evaluation set in parts, one isolated subprocess at a time. The following are strictly prohibited and lead to disqualification:

Producing or selecting a prediction using information from evaluation windows other than the one it answers.
Reading evaluation data files from the container filesystem instead of, or in addition to, the values passed through the interface.
Encoding evaluation data into prediction outputs, logs, or any other channel that leaves the container.
Returning lower / upper intervals estimated from the input window you were given is what sps_score rewards, and is unaffected by the first item.

Suspicious score patterns are flagged and manually reviewed.

External Resources
To ensure fairness, participants are not allowed to generate additional data for training, nor are they allowed to rely on pretrained models trained on anything other than the competition's released training data. Checkpoints trained from that release are permitted, including the official baselines the organizers provide. This will be guaranteed during the final retraining evaluation stage. See the FAQ for how this applies to standard data augmentation.

Participants may use open-source software when allowed by its licenses. Any external resource used for model development should be disclosed in the final report or fact sheet when requested by organizers.

During official Codabench evaluation, submissions must not rely on private external services, participant-controlled API keys, or network calls. All files required for inference should be included in the submitted archive or be part of the approved starting kit environment.

Reproducibility
Winning teams must provide complete code and instructions sufficient for organizers to reproduce their solution. To be eligible for prizes, winning teams must agree to open-source their complete solution before the NeurIPS results presentation event.

Final Verification
Final rankings are determined in the organizer-controlled Decision Phase, not by the development leaderboard. At the end of the Development Phase the top 10 teams on the Track 1 leaderboard are shortlisted. Organizers re-train each shortlisted team's method from scratch on the competition's GPU cluster to ensure reproducibility, then evaluate on a private test set with unseen parameter regimes (e.g., novel angles of attack or Reynolds numbers); the resulting private-test scores decide the top 3. Organizers also verify code integrity, check rule compliance, and run robustness tests. Suspicious score patterns are flagged and manually reviewed. A shortlisted solution that cannot be reproduced through re-training, or that shows evidence of improper use of evaluation data, is disqualified and replaced by the next-ranked team.

The exact final submission package format (training code layout, environment specification, and an accompanying spec document) will be announced on the FAQ page about 5 days before the end of the Development Phase.

Ranking
Track 1 ranking is determined by final_score on the official Sim2Real leaderboard. Organizers reserve the right to audit top submissions for rule compliance and reproducibility.

Rights Reserved
Final placements are decided by the Decision Phase procedure described under Final Verification, not by the development leaderboard and not at anyone's discretion. Reviewing a shortlisted team's code before confirming a placement is part of that procedure.

Where the procedure does not settle a question, the organizers decide, and will publish the reasoning alongside the result.

Participants agree under the Terms page not to acquire, redistribute or retain the evaluation data outside the competition. The organizers reserve the right to pursue remedies for a breach of that undertaking.

The Terms page also sets out what the organizers may do with a submission, and what may be changed during the competition.


FAQ
FAQ
Q: What is the submission size limit? A: The extracted submission archive, model checkpoint included, must stay under 256 MB. Oversized submissions fail immediately. See the Submission page.

Q: What is the execution time limit? A: 5 minutes of container execution per submission in the Warm-up and Development phases (data download does not count). Exceeding it marks the submission Failed with no score. The official Transolver baseline runs in about 1 minute.

Q: My submission failed with "Transient platform storage error while downloading the 'submission' bundle". What happened? A: This is a platform-side (Codabench) storage issue, not a problem with your submission. When the platform's object storage stalls while the evaluation worker fetches your archive, the submission is marked Failed. Failed submissions do NOT count against your quota: refresh the page and you will see the attempt returned. Please resubmit, ideally about an hour later so the platform has time to recover.

Q: Can I install extra packages (requirements.txt)? A: No. Evaluation runs fully offline and nothing is installed. Use the libraries listed on the Submission page (Evaluation Environment), and vendor pure-Python packages inside your archive if needed.

Q: Is model/checkpoint loading time counted in the Time score? A: No. Model construction and checkpoint loading run outside the timed region; only inference is timed. Note the platform-level execution time limit (wall clock for the whole run) does include loading, so keep it reasonable.

Q: Does "not allowed to generate additional data for training" ban standard data augmentation? A: No. Standard augmentation of the released data is fine in both tracks: anything that takes a released sample and produces a variant of it on the spot during training. What we check is provenance. Every training sample must trace back to the released training data through a transform you can point at. Extra simulations, external datasets, and pretrained models from other sources stay out. The reason is fairness at the final retraining stage. Shortlisted methods are re-trained from scratch and organizers hold only the released training data, so we accept training code only: no new simulation data, no code that runs new simulations from scratch, no model weights, and no pretrained models from other sources. If your training cannot be reproduced from the release alone, it cannot survive that stage. Augmentation code ships with your training code and runs as part of training.

Q: How are final rankings determined? A: The top 10 teams on the development leaderboard are shortlisted; organizers re-train their methods on the competition's GPU cluster and evaluate on a private test set with unseen parameter regimes, and those private-test scores decide the top 3. See Rules → Final Verification.

Q: How and when do finalists submit their training code? A: The exact final submission package format (training code layout, environment specification, and an accompanying spec document) will be announced on this page about 5 days before the end of the Development Phase. Keep your training pipeline reproducible from the start so the package is easy to assemble.

Q: What timezone are the phase deadlines in? A: All phase boundaries are in UTC. Each phase starts at 00:00:00 UTC on its start date and ends at 23:59:59 UTC on its end date. Deadlines are NOT Anywhere-on-Earth.

Q: My submission fails with a permission error inside DataLoader. Why? A: Your model runs in an isolated subprocess and worker processes are not available, so torch.utils.data.DataLoader must use num_workers=0. That is the default, so this only affects submissions that set it explicitly. The error surfaces as PermissionError: [Errno 13] and can be followed by an unrelated-looking AttributeError from PyTorch's own shutdown path.

Q: Where do I ask questions? A: The competition forum, or realpde-competition@googlegroups.com.




Terms And Conditions
By registering for this competition, participants agree to follow the official rules and to use the provided data, code, and evaluation platform only for research and competition purposes.

Data Use
The RealPDE competition data is provided for non-commercial research and competition participation. Participants must not redistribute hidden validation or test data, attempt to reverse engineer private labels, or use the data in a way that violates the data license.

Fair Evaluation
Participants must not exploit implementation bugs, access hidden reference data, or use multiple accounts to bypass submission limits. Submissions may be disqualified if they violate the competition rules or compromise the integrity of the evaluation.

Your Submissions
Every submission is stored and run by the organizers for as long as the competition and its results paper need it. Within that, the organizers may run it, re-run it, score it, inspect its code, and describe its approach and its scores in the results paper and in public rankings.

Submitting does not transfer ownership and does not give the organizers a licence to release your code or weights. A submission that does not win is not published by the organizers, and is not shared outside the organizing team.

Code Release For Winners
Prize eligibility requires winning teams to provide a reproducible solution and agree to open-source their complete code before the NeurIPS results presentation event. What licence that release carries is the winning team's own choice.

Organizer Rights
Organizers may update documentation, fix evaluation bugs, audit submissions, rerun submissions, or adjust administrative details when needed to preserve fairness and reproducibility. Any material change affecting ranking will be announced to participants.

How final placements are decided, and what happens when a question the procedure does not cover comes up, is on the Rules page under Final Verification and Rights Reserved.