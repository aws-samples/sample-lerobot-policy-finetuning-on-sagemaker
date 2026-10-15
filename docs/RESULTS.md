# Results and validation ledger

Every number in this file comes from a recorded run, and nothing is re-derived. Unless a row says otherwise, the run was a
SageMaker job of this pipeline. Two groups of rows are different. The two reference runs predate this repository: one ran
on an EC2 L40S host, the other as a SageMaker job on ml.g6e.16xlarge, both with the same ACT recipe. The four 80 GB smokes
ran on EC2 H100 hosts with this repository's image, input channels and hyperparameters (see the note under the smoke
table). Dates are UTC.

The files that take part in an ACT job (Dockerfile, lock files, entrypoint, trainer scripts, `act.yaml`) are listed and
hashed in `tests/test_frozen_set.py`. When any of them changes, the ACT result has to be re-checked: the hash in that test
is updated in the same change, a row is added to the last table below, and the next release image is run through the
20-step CPU ACT smoke (`scripts/smoke_aws.sh`).

"x hold" is the error of the policy divided by the error of a "hold still" baseline. The error is the mean absolute error
(MAE) between the action chunk the policy predicts from one observation and the recorded actions, measured on held-out
episodes the policy never trained on. The baseline predicts that the arm stays where it is for the whole chunk. Below 1.0
the policy has learned something. The baseline grows with the chunk length, so ratios are comparable only between runs
with the same chunk.

## Reference runs (recorded before this repository, private reference dataset)

| run | where | held-out joints MAE | x hold | grippers MAE | x hold | gripper timing | dims beating hold |
|---|---|---|---|---|---|---|---|
| ACT, 100k steps, 2026-08-18 | EC2 L40S, 2 h 37 m | 0.04323 | 0.679 | 0.00158 | 0.404 | 16/16 | 14/16 |
| ACT on SageMaker, 100k steps, 2026-09-09 | ml.g6e.16xlarge, 2 h 50 m | 0.04271 | 0.671 | 0.00160 | 0.408 | 16/16 | 14/16 |

Dataset: an 80-episode bimanual pick-and-place recording (LeRobot v3.0, 3 cameras 720x1280, 25 fps, 16 action dims).
Held-out episodes 0 13 26 30 43 56 69 79, evaluated at stride 1 (every held-out frame); the other 72 episodes (21,557
frames) trained. The hold baseline on this split is 0.06367 rad on joints and 0.00393 rad on grippers. The dataset is not
distributed with this repository. The acceptance band in `pipeline/reference_results.json` (key `pick-and-place-v3.0`) is
one-sided (upper bounds on every metric, derived from these two runs) and applies only to that dataset. On any other
dataset the verdict is `n/a`. Reference rows are printed only when `pipeline/reference_results.json` has an entry for that
policy and dataset name (today `svla_so101_pickplace` for act, smolvla, xvla, pi0 and pi05); otherwise the report table
shows this run alone.

**Re-validation through this pipeline** (2026-09-10, same dataset, 100k steps, ml.g6e.4xlarge on-demand, release image):
two runs of 11471 s and 11438 s billable (about $17 each). The first run got the verdict REPEAT: its joints error was inside
the pass bound (0.04315 = 0.678 x hold), but only 13 of 16 dimensions beat the hold baseline, and the missing one sat within
5 % of it (1.02 x hold). `pipeline/report.py` asks for one repeat in that case. The repeat gave joints 0.04297 = 0.675 x hold,
grippers 0.00160 = 0.409 x hold, gripper timing 16/16 and 14/16 dimensions beating hold: **PASS**. The SageMaker path
reproduces the validated ACT result. A 200-step GPU smoke on the same image ran at 10.0 steps/s with 4.27 GB of GPU memory.

## Full fine-tunes on the public dataset

All runs used one `ml.g6e.4xlarge` (L40S 48 GB) with network isolation on, and every published checkpoint passed the
reload check. Dataset: `lerobot/svla_so101_pickplace` @ `v3.0` (50 episodes, 2 cameras 480x640 downscaled to 360 px
height, 30 fps, 6 joints in degrees), held-out episodes 0 10 20 30 40. ACT was evaluated at stride 1 (every held-out
frame); the VLAs at stride 10 with 4 samples per frame. "Staged" means a CPU job downloaded the base weights into the
bucket beforehand, so the training job ran without network access. Costs are the on-demand list prices of the region the
runs were recorded in (Asia Pacific, Tokyo) at the time of the run.

| date | policy | steps x batch (chunk) | billable | cost | joints MAE (x hold) | dims beating hold | step rate |
|---|---|---|---|---|---|---|---|
| 2026-09-10 | `act` (public reference run) | 100k x 8 (30) | 6822 s | ~$10 | 4.305 (**0.424**) | 6/6 | 14.7 steps per billable second |
| 2026-09-11 | `act`, repeated from a fresh clone by following the README quick start, after the SageMaker execution role had been reduced to read and put on the bucket plus delete under `jobs/` and `hub/` only (docs/SECURITY.md) | 100k x 8 (30) | 6743 s | ~$10 | 4.333 (0.427) | 6/6 | 14.8 steps per billable second; its 200-step smoke took 436 s; 20 checkpoints and the outputs synced, no S3 access errors |
| 2026-09-14 | `act` on **managed Spot** (`--spot`) | 100k x 8 (30) | 3276 s of 6741 s training (**51 % saved**) | ~$5 | 4.274 (0.421) | 6/6 | no interruption; maximum wait set to twice the runtime cap |
| 2026-09-11 | `smolvla` (action expert trained from scratch on the staged SmolVLM2 backbone) | 20k x 16 (50) | 4960 s | ~$7 | 5.458 (0.376) | 6/6 | 4.58 steps/s steady (96 % of its smoke rate), 4.03 per billable second |
| 2026-09-11 | `xvla` (fine-tune of the staged `lerobot/xvla-base`; licence class `terms`, acknowledged with `accept_licence: true`) | 30k x 16 (24) | 13881 s | ~$20 | 4.031 (0.463) | 6/6 | 2.25 steps/s steady (99 % of its smoke rate), 2.16 per billable second |
| 2026-09-11 | `pi0` (staged `lerobot/pi0_base`, vision encoder frozen and only the action expert trained; licence class `terms`) | 20k x 8 (50) | 17479 s | ~$26 | 5.400 (0.372) | 6/6 | 1.23 steps/s steady (97 %), 1.14 per billable second |
| 2026-09-11 | `pi05` (staged `lerobot/pi05_base`; the q01/q99 quantile statistics it needs were computed inside the job; licence class `terms`) | 20k x 8 (50) | 22906 s | ~$34 | 5.074 (0.350) | 6/6 | 0.93 steps/s steady (97 %), 0.87 per billable second |
| 2026-09-21 | `fastwam_lora` (the shipped default: LoRA on the video expert of the staged `lerobot/fastwam_base`, action expert and proprio encoder in full training; run on an EC2 p5.4xlarge Capacity Block with the SageMaker container contract, image `sha256:65bd643b4034`) | 20k x 1 (30) | 17421 s of host time (4 h 50 min incl. eval, the CPU adapter merge and the merged-copy reload), not a SageMaker bill | the Capacity Block is prepaid; on `ml.p5.4xlarge` on-demand this would be ~$40 | 53.03 (**8.85**) as measured, but see note | 0/6 | 1.21 steps/s steady (99 % of its smoke rate); loss 1.87 -> 0.37; peak 74.7 GB. Batch 1 x 20k steps is 0.08 epochs of the 4-task set: the recipe runs end to end (adapter 3.9 GB, merged 23 GB bit-identical to adapter + base, `inference/predict.py` serves the merged copy on the GPU in 2 min) but the default budget is far too small to learn the task; the 2026-09-18 30k run that held 0.36-0.42 x hold started from an earlier full fine-tune (advanced flow). **Corrected 2026-09-22:** the recipe never trained the action encoder and head (PEFT wrapped the whole action expert while FastWAM's forward uses an unregistered alias to the original; their weights are exactly at the default init), and eval and reload scored a random head that differs per process, hence 8.85 x hold; the merged copy, which carries the saved copy of that untrained head, re-evaluates at 12.997 (2.17 x hold) on the same episodes. Recipe fixed the same day (leaf-level `full_training_modules`); the rerun with the fixed recipe is the next `fastwam_lora` row. Status stays *smoke-tested* |
| 2026-09-22 | `fastwam_lora` with the fixed recipe (leaf-level `full_training_modules`: the action encoder and head now train; same host, image `sha256:ba67992091ad`) | 20k x 1 (30) | 17500 s of host time (4 h 52 min incl. eval, the CPU adapter merge and the merged-copy reload), not a SageMaker bill | prepaid Capacity Block; ~$40 on `ml.p5.4xlarge` on-demand | 12.03 (**2.01**) | 0/6 | 1.22 steps/s steady; loss 1.41 -> 0.38; peak 70.6 GB; `action_encoder` and `head` moved well beyond their init (max 0.479 vs 0.408; head std 0.0051 from 0.018); adapter 3.9 GB, merged 23 GB; reload of the merged copy EXACT (diff 0.0). Four times better than the broken recipe's 8.85 x hold, but still twice the hold-still error: 20k steps at batch 1 is 0.08 epochs of the 4-task set, so the shipped budget under-trains the recipe (the 2026-09-18 30k advanced-flow run reached 0.36-0.42 x hold from an earlier fine-tune). Status stays *smoke-tested*; raising `steps` (and `batch_size` where memory allows) is the next experiment |
| 2026-09-23 | `fastwam_lora`, fixed recipe, budget experiment: 3x the shipped steps (same host, image `sha256:ba67992091ad`) | 60k x 1 (30) | 51600 s of host time (14 h 20 min incl. two evals, the CPU merge and the reload), not a SageMaker bill | prepaid Capacity Block; ~$120 on `ml.p5.4xlarge` on-demand | 7.55 (**1.26**) | 2/6 | 1.20 steps/s steady; loss 0.30 at 60k (0.24 epochs); the 40k checkpoint scored 7.95 (1.33 x hold), so the curve is 20k 12.03 -> 40k 7.95 -> 60k 7.55: still above hold-still and flattening; peak 70.6 GB; adapter 4.18 GB, merged 24.08 GB; reload of the merged copy EXACT (diff 0.0). Status stays *smoke-tested*: the recipe learns, but on this dataset it has not beaten hold at 3x the default budget; a larger batch (96 GB) or a higher learning rate is the next lever |

What the four VLA runs taught about sizing: in steady state they held 96-99 % of the step rate their smoke had measured,
and delivered 85-95 % of it per billable second (start-up, evaluation and checkpoint writes account for the rest). They
used 59-69 % of their runtime caps, which `examples/project_config.vla.json` sets with `runtime_multiplier: 1.5`. The
shipped default of 3.0 is headroom for a slower fallback instance or a heavier dataset, not a measured slowdown. A
pi0-class checkpoint is 16.1 GB of fp32 weights plus 2.5 GB of optimizer state. With twenty of them kept, one run holds
about 370 GB in S3, which is why `compute.keep_checkpoints` defaults to 3. `model.tar.gz` for the pi0 family is 8.4-8.8 GB
compressed and unpacks to about 16 GB. The VLAs' x-hold ratios are not comparable with ACT's 0.424 because the chunk
lengths differ (24 and 50 frames against ACT's 30; see the note on "x hold" above). [docs/WHY_NOT_VLA.md](WHY_NOT_VLA.md)
discusses the head-to-head comparison.

**Resume and extend** (2026-09-10): `--resume-from` the finished 100k re-validation job with `--extend-steps 100400` restored
the 20 checkpoints from S3, repaired the `last` symlink, trained the extra 400 steps, evaluated and published (at stride 60,
so the numbers carry no verdict). A Spot interruption takes the same path (checkpoint restore from S3, symlink repair, resume
from the newest complete checkpoint), so this run is the evidence that Spot resume and `--extend-steps` work.

## 200-step smokes (the gate that promotes a registry entry to `smoke-tested`)

The smoke trains 200 steps on the same instance type and image as the training job, then applies the gate defined in
[docs/ARCHITECTURE.md](ARCHITECTURE.md) (smoke row). The step rate it measures sets the runtime cap of the training job.
After 200 steps most policies are still worse than the hold baseline; that is expected and not part of the gate.

| date | policy | dataset (see docs/DATASETS.md) | instance | steps/s at batch | gate | notes |
|---|---|---|---|---|---|---|
| 2026-09-14 | `act` | svla_so101_pickplace | ml.g6e.4xlarge | 16.67 at 8 | PASS | 0.72 x hold after 200 steps; 436 s billable (GPU memory 4.27 GB, measured on the reference dataset, see above) |
| 2026-09-10, re-smoked 2026-09-14 | `diffusion` | svla_so101_pickplace | ml.g6e.4xlarge | 6.25 at 16 | PASS | the evaluation feeds it an observation history of 2 frames |
| 2026-09-11, re-smoked 2026-09-14 | `vqbet` | svla_so101_pickplace, one-camera view | ml.g6e.4xlarge | 5.40 at 16 | PASS | lerobot's VQ-BeT accepts one camera, so the job builds a one-camera view (`select_camera.py`); the smoke passes `--policy.n_vqvae_training_steps=100` so both training phases run within 200 steps |
| 2026-09-14 | `tdmpc` | xarm_lift_medium | ml.g6e.4xlarge | 2.94 at 256 (registry default) | PASS | the only public v3.0 dataset with reward labels; its 15 fps timestamps need `--tolerance_s=0.002` (in the recipe) |
| 2026-09-14 | `multi_task_dit` | svla_so101_pickplace | ml.g6e.4xlarge | 7.14 at 8 | PASS | images are normalised with the dataset's mean and standard deviation (MEAN_STD), so the dataset needs sound image statistics; CLIP encoder staged |
| 2026-09-10 and 2026-09-14 | `smolvla` | svla_so101_pickplace; so101-4tasks-100eps | ml.g6e.4xlarge | 4.77 and 4.55 at 16 | PASS | weights kept in fp32 with bf16 autocast (`trainer/amp_driver.py`); the gate's check for bf16-stored trainable weights reads the dtypes recorded in the saved safetensors file, not the in-memory ones |
| 2026-09-10 and 2026-09-14 | `xvla` | svla_so101_pickplace; so101-4tasks-100eps | ml.g6e.4xlarge | 2.77 and 2.27 at 16 | PASS | the policy samples its output, so the reload check compares averages of 4 draws |
| 2026-09-11 and 2026-09-14 | `pi0` | svla_so101_pickplace; so101-4tasks-100eps | ml.g6e.4xlarge | 1.25 and 1.23 at 8 | PASS | the staging job copied `lerobot/pi0_base` and the gated PaliGemma tokenizer (Hugging Face token read from Secrets Manager) in about 8 min |
| 2026-09-11 and 2026-09-14 | `pi05` | svla_so101_pickplace; so101-4tasks-100eps | ml.g6e.4xlarge | 0.95 and 0.96 at 8 | PASS | the q01/q99 quantile statistics were computed inside the job (`ensure_quantiles.py`) |
| 2026-09-14 | `evo1` | so101-4tasks-100eps | ml.g6e.4xlarge | 1.37 at 8 | PASS | `OpenGVLab/InternVL3-1B-hf` staged; licence class unknown, accepted for the smoke only |
| 2026-09-14 | `molmoact2` | so101-4tasks-100eps | ml.g6e.4xlarge | 0.32 at 8 | PASS | 21.8 GB base staged; smoke checks run on CPU; 150 min smoke cap; the research-only licence was acknowledged for this smoke with `accept_licence: true` and an `approval_ref` in the run's `project_config.json` |
| 2026-09-14 | `groot` | svla_so101_pickplace | ml.g6e.4xlarge (48 GB sufficed) | 2.17 at 8 | PASS | `nvidia/GR00T-N1.7-3B` staged complete (`ignore_patterns: []`) plus the gated Cosmos processor files |
| 2026-09-14 | `lingbot_va` (LoRA r16) | so101-4tasks-100eps | ml.g6e.4xlarge | 0.11 at 1 | PASS | the UMT5-XXL prompt encoder runs on CPU; the base was trained on LIBERO's cameras and a 14-dim action, so the job empties its input and output features (`base_config_patch`) and passes the dataset's cameras; the LoRA adapter is merged before the reload check |
| 2026-09-15 | `pi0_fast` | so101-4tasks-100eps | 80 GB H100 (see note) | 1.61-1.67 at 4 (chunk 10) | PASS twice | peak 68.1 GB: a 48 GB L40S cannot run it; the FAST action tokenizer cannot be found by repository id when the job is offline, so the registry points its loader at the staged snapshot directory (`train.local_repo_args`) and lists a text replacement for the base's preprocessor JSON (`base_patches`); trainer/policies/README.md explains both |
| 2026-09-15 | `eo1` | so101-4tasks-100eps | 80 GB H100 (see note) | 0.55 at 4 (chunk 30) | PASS | peak 74.2 GB (no headroom for batch 8); `--policy.use_fast_processor=true` is required under `amp_driver.py`; [docs/TRAPS.md](TRAPS.md) has the row on the transformers processor_kwargs warning |
| 2026-09-15 | `fastwam` | so101-4tasks-100eps | 80 GB H100 (see note) | 2.25 at 1 (chunk 30) | PASS | peak 72.8 GB; action and state widths are taken from the dataset. `hardware.checkpoint_gb` is 31 (23 GB weights + 7.7 GB optimizer state for the action expert), measured on a later 20k-step FastWAM fine-tune on a private 16-dim bimanual dataset that used the same image, channels and hyperparameters as a SageMaker job but is not listed in this file; the job volume derived from it is about 230 GB |
| 2026-09-18 | `fastwam_lora` | a private 16-dim bimanual dataset (the same recording as the `fastwam` checkpoint note above) | 80 GB H100 (see note) | 0.95 at 1 (chunk 32) | PASS | peak 78.8 GB (gradient checkpointing on, backprop through the 5B video expert); adapter checkpoint 4.2 GB (`adapter_model.safetensors`, no `model.safetensors`); LoRA rank 16 on the video expert attention with the action expert and proprio encoder in full training; video loss on (`lambda_video` 1.0); trained from an earlier frozen-video-expert `fastwam` fine-tune passed as `--policy.pretrained_path` (the advanced flow in [docs/POLICIES.md](POLICIES.md), LoRA recipes). Reload check ok, weights-moved gate passed on the adapter's `lora_B` tensors. A 30k-step run with the same recipe finished (rc 0) and held its held-out joints ratio between 0.36 and 0.42 x hold from 5k steps on |
| 2026-09-21 | `fastwam_lora` | so101-4tasks-100eps (the shipped default: `lerobot/fastwam_base` from the hub channel) | EC2 p5.4xlarge (1x H100 80 GB, an EC2 Capacity Block; the SageMaker container contract reproduced on the host: same image `sha256:65bd643b4034`, `train` + `hub` channels, hyperparameters the pipeline submits for this policy) | 1.22 at 1 (chunk 30) | PASS | peak 74.7 GB; adapter checkpoint 4.18 GB; adapter merged into `pretrained_model_merged/` (24.08 GB `model.safetensors`, bit-identical to adapter + base merged in memory: 1651 tensors, max abs diff 0.0); reload check of the merged copy passed only in stochastic mode (diff 209 vs the eval; two reloads of one checkpoint differed by 86 and 56, reloads of the merged copy by 209-251). Explained 2026-09-22: not sampling but the untrained action encoder and head reached through FastWAM's alias, whose base-side random init differs per process (docs/TRAPS.md); the fixed recipe reloads exactly, next row. Total 18 min per run. `inference/predict.py --device cuda` loads the merged copy and returns the 32 x 6 chunk (3 min 40 s including the 24 GB copy it makes to patch the config). The first run on `sha256:ce006414` found the merge writing its scratch outside `/opt/ml/model` (`EXDEV` on the mount) and the reload check failing on sampling noise; both fixed the same day (docs/TRAPS.md). |
| 2026-09-22 | `fastwam_lora` (recipe fixed: leaf-level `full_training_modules`) | so101-4tasks-100eps (the shipped default) | EC2 p5.4xlarge (as above; image `sha256:ba67992091ad`) | 1.21 at 1 (chunk 30) | PASS | peak 70.6 GB; the action encoder and head now train (after 200 steps their weights exceed the default-init bound: 0.41091 > 0.40825, 0.03151 > 0.03125; on the 2026-09-21 20k run they never moved); adapter merged; `eval_openloop.load_policy` rebuilds the merged policy from its state dict, so the reload of the merged copy reproduces the eval exactly (diff 0.0, strict mode); `reload_check.py` symlinks the checkpoint instead of copying 24 GB into /tmp (the copy filled the container filesystem on the first attempt) |
| 2026-09-21 | `pi0_fast` | so101-4tasks-100eps (staged `lerobot/pi0fast-base` + PaliGemma tokenizer from the hub channel) | EC2 p5.4xlarge (1x H100 80 GB, an EC2 Capacity Block; SageMaker container contract on the host, release image `sha256:65bd643b4034`) | 1.67 at 4 (chunk 10) | PASS | first run of the release image itself (the 2026-09-15 row used an older image plus trainer patches); peak 68.1 GB; weights moved 1.0, no bf16 trainable params, reload check exact (diff 0.0); 18 min end to end |
| 2026-09-21 | `eo1` | so101-4tasks-100eps (staged `Qwen/Qwen2.5-VL-3B-Instruct` from the hub channel) | EC2 p5.4xlarge (as above) | 0.56 at 4 (chunk 30) | PASS | release image; peak 74.2 GB (6 GB headroom, as on 2026-09-15); reload ok in stochastic mode (`n_samples` 4, diff 30.8 informational); 16 min end to end |
| 2026-09-21 | `fastwam` | so101-4tasks-100eps (staged `lerobot/fastwam_base` + Wan2.2 pieces from the hub channel) | EC2 p5.4xlarge (as above) | 2.22 at 1 (chunk 30) | PASS | release image; peak 72.8 GB; weights moved 1.0; reload check EXACT (diff 0.0): the flow-matching draw is deterministic, which pointed at the LoRA recipe for its reload spread (found 2026-09-22: the untrained action head, see the `fastwam_lora` rows); 25 min end to end |

Note on the 80 GB rows: the four 80 GB smokes (`pi0_fast`, `eo1`, `fastwam`, `fastwam_lora`) ran on
EC2 H100 hosts with the release image plus the trainer changes that went into the next release image (`local_repo_args`,
`base_patches`, `eval.no_history` and the dataloader worker cap), the same `train` and `hub` channels, `HF_HUB_OFFLINE=1`
and the same hyperparameters as a SageMaker job. The gate, the step rates and the memory peaks are those runs. A SageMaker
job with an 80 GB quota runs the identical container and is expected to reproduce them.

What `fastwam_lora` is next to `fastwam` is described in [docs/POLICIES.md](POLICIES.md), LoRA recipes. The open-loop
held-out ratio did not separate the two recipes, so treat it as a training signal, not as a measure of task success.

## Release image

The release image (`image.tag` `v1`) is rebuilt by CodeBuild whenever `trainer/` changes; a build for an unchanged
`trainer/` tree re-tags the existing image (same digest, no second build). Every rebuild is followed by
`scripts/smoke_aws.sh`, a 20-step CPU ACT job on the published image. The script asserts that the job completes and
publishes a finite joints MAE; with `--expect-joints-mae 17.3817` it also fails unless the MAE is the value every
validated build has produced on the public dataset, which shows that the ACT path is unchanged. From 2026-09-16 on, the
evidence column of the table below names the digest each change was validated with (earlier rows record the smoke outcome
only); the digest your own jobs use is the value of the SSM
parameter `/<project>/image_uri` in your account. The image's package set is fixed by lock files
([docs/ARCHITECTURE.md](ARCHITECTURE.md), "Image"). Building it from scratch takes about 14 min and yields a 13.5 GB
compressed image in ECR (about 26 GB unpacked).

## Changes to the frozen ACT files after the freeze

Each row is a change to one of the files hashed in `tests/test_frozen_set.py` (the files that take part in an ACT job),
whether it can affect the ACT path, and the evidence that closed it. The current hash is in that test.

| date | change | ACT path affected? | evidence |
|---|---|---|---|
| 2026-09-10 | hub staging: the entrypoint exports `HF_HOME` / `HF_HUB_OFFLINE` when a `hub` channel is mounted | no (inert without the channel) | CPU smoke of the next image |
| 2026-09-10 | precision launcher: `train_policy.sh` runs `amp_driver.py` for `fp32_masters_bf16_autocast` policies; ACT keeps `lerobot-train` | no (rendered command line unchanged, checked by a test) | CPU smoke of the next image |
| 2026-09-10 | evaluation: observation history (diffusion), batch axis for history policies, reload check for policies that sample their output (X-VLA), bf16 check on saved dtypes (SmolVLA) | no (ACT: no history, one sample, no bf16) | diffusion, SmolVLA and X-VLA smokes PASS |
| 2026-09-10 | verification fixes: resume rendering drops `--policy.*` overrides, height gates accept `[3,H,W]` shapes, a single all-copy data file fails the check, checkpoint selection requires `train_config.json` | ACT: resume rendering and checkpoint selection only | release image rebuilt, CPU ACT smoke PASS |
| 2026-09-11 | camera selection for single-image policies (`select_camera.py`, `camera` hyperparameter) | no (`single_camera_required` false for ACT) | VQ-BeT smoke PASS; CPU ACT smoke PASS |
| 2026-09-11 | smoke reuse requires a passed gate; fallback instances re-plan the runtime cap; VQ-BeT smoke-only flags and queue-fed evaluation; resume picks complete saves only | ACT: stricter resume filter only | CPU ACT smoke PASS |
| 2026-09-11 | per-policy dataset override; quantile statistics completed in a stats view for `requires_quantile_stats` policies | no (ACT uses the shared dataset) | pi0.5 smoke PASS; CPU ACT smoke PASS |
| 2026-09-14 | `summary.py` step rate parses lerobot's rounded `step:NK` labels (before this fix the SUMMARY.json of every run longer than 1000 steps reported `step_rate: 0.0`, because lerobot rounds the step label to whole thousands and the parser saw no progress); the licence notice and SUMMARY.json carry the dataset fingerprint, image digest, run status and licence-gate outcome; tokenizer roles stage only tokenizer files | no (ACT command line unchanged) | release image rebuilt, CPU ACT smoke PASS |
| 2026-09-14 | generalised evaluation protocol for the remaining policies (queue keys, chunk-first, no-history, PEFT adapter checkpoints, `reset_per_call`, `tolerance_s`); smoke cap and smoke-check device from the registry | no (all new keys default off for ACT) | tdmpc, multi_task_dit, evo1, molmoact2, groot, lingbot_va smokes PASS; CPU ACT smoke PASS |
| 2026-09-15 | checkpoint retention: the entrypoint prunes to `compute.keep_checkpoints` complete checkpoints every 2 min and at the end (the S3 mirror follows); content-hash image tag skips rebuilds of an identical tree | ACT: only older complete checkpoints are deleted; publish and resume read the newest | release image rebuilt, CPU ACT smoke PASS |
| 2026-09-15 | stats view for every policy (ImageNet-shaped placeholders for cameras missing from `stats.json`); camera subsets (`policies.<name>.cameras`) | no (inert for a complete `stats.json` and without a `cameras` list) | release image rebuilt twice, CPU ACT smokes PASS |
| 2026-09-16 | publish clean-up: the `seed` hyperparameter is honoured (its export had been trapped inside a shell comment in `train_entrypoint.sh`, so every job trained with the fallback 1000, which is also the shipped default); comment and docstring wording only, plus scanner-suppression comments in the Dockerfile and `image_asserts.py` (no behaviour change) | ACT: none when `seed` is left at 1000 (the validated value); a user-set seed now takes effect | release image `v1` rebuilt as `sha256:2cc7907c7dd2`, CPU ACT smoke PASS (2026-09-16, 340 s billable, joints MAE 17.3817 unchanged) |
| 2026-09-16 | base-checkpoint patch hook: `train_policy.sh` now runs `patch_base_config.py` when a registry entry has `base_patches` but no `base_config_patch` (pi0-FAST's tokenizer path rewrite; before this fix only `base_config_patch` triggered it) | no (ACT has neither key and no `hub` channel) | release image `v1` rebuilt as `sha256:2cc7907c7dd2`, CPU ACT smoke PASS (2026-09-16, 340 s billable, joints MAE 17.3817 unchanged) |
| 2026-09-18 | optional adapter merge in `publish.py` (`publish.merge_adapter` registry flag or `merge_adapter` hyperparameter): a PEFT adapter checkpoint is merged into full weights under `pretrained_model_merged/` with a servable config, `summary.py` records the outcome, `reload_check.py` prefers the merged copy when it exists, `train_entrypoint.sh` passes the hyperparameter through | ACT: none (a full checkpoint with no flag takes no new path apart from removing a `pretrained_model_merged/` directory that cannot exist) | release image `v1` republished 2026-09-21 as `sha256:ce006414b33c` (`build_image --tag v1 --publish-ssm`; the content-hash tag re-used the 2026-09-18 dev build of the same tree, so the build took 2 min), CPU ACT smoke PASS (a 20-step CPU ACT smoke, 350 s, joints MAE 17.3817 unchanged); hash `96994b9c` in `tests/test_frozen_set.py`, previous `583b406a` |
| 2026-09-21 | `publish.py` builds the LoRA merge scratch as a hidden sibling inside the model directory (same filesystem as `/opt/ml/model`, which is its own mount; the earlier sibling-of-model scratch failed the `os.replace` with `EXDEV`); `train_entrypoint.sh` passes `--stochastic` to `reload_check.py` when the registry entry has `eval.stochastic: true` (fastwam, fastwam_lora) | ACT: none (the merge runs only for adapter checkpoints; the flag is read from the registry and is absent for ACT) | release image `v1` republished as `sha256:65bd643b4034` (an intermediate `sha256:374ca2e9948d` carried the merge fix alone), CPU ACT smoke PASS (a 20-step CPU ACT smoke, 350 s, joints MAE 17.3817 unchanged); fastwam_lora p5 smoke PASS on it (smoke table); hash `74c9199c` in `tests/test_frozen_set.py`, previous `96994b9c` |
| 2026-09-22 | `trainer/policies/fastwam_lora.yaml`: `full_training_modules` names the action expert's leaf modules instead of the whole expert (PEFT put the trainable copy where FastWAM's forward never looks); `eval_openloop.load_policy` rebuilds the merged PEFT policy from its state dict on the target device; `reload_check.py` and `inference/predict.py` symlink a checkpoint to patch its config instead of copying it | ACT: none (no adapter path; the symlink view replaces a copy of the same files) | release image `v1` republished three times while the fix was validated (`sha256:e28dfccf9839`, `sha256:baca7280738d`, final `sha256:ba67992091ad`), CPU ACT smoke PASS on each (three 20-step CPU ACT smokes, 340 s, 350 s and 351 s, joints MAE 17.3817 unchanged); fastwam_lora smoke PASS on the final image with an exact reload (smoke table); hash `e6b0105c` in `tests/test_frozen_set.py`, previous `74c9199c` |
| 2026-09-23 | build-time asserts: `image_asserts.py` imports every non-stub registry policy class (eo1, fastwam, lingbot_va and groot added; groot is a hard check, its class imports without decord); the Dockerfile verifies the pre-staged ResNet18 weights by their full sha256 instead of the file size; the `eval.stochastic` comment in `train_entrypoint.sh` describes the option instead of the fixed defect; `pipeline.smoke_aws` gains the opt-in `--expect-joints-mae` fingerprint check | ACT: the weight-file check only (same file, stricter test); the comment is inert | release image `v1` rebuilt as `sha256:d03dd0797805` (CodeBuild, lock mode enforce, published to SSM), CPU ACT smoke PASS on it with `--expect-joints-mae 17.3817` (2026-09-23, 344 s billable, joints MAE 17.381718, within 1.8e-5 of the fingerprint); hash `6a5dad9c` in `tests/test_frozen_set.py`, previous `e6b0105c` |
