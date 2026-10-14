# Testing this repository

This page is for anyone who must confirm that a checkout works: a maintainer checking a release candidate, a reviewer
with a fresh clone, or the first person to use this repository in a new AWS account. The checks come in three layers.
Each layer shows more and costs more. Layer 1 needs a laptop and nothing else. Layer 2 is the README quick start on the
shipped public dataset, with what a healthy run printed, how long it took and what it cost (about $11 in total). Layer 3
is what you can do with the result. Commands, flags, paths and field names below are copied from the code. If one does
not match what you see, the code has moved on and this page needs an update.

## 1. Laptop only (no AWS credentials, no Docker, no GPU)

### One command

```bash
bash scripts/verify_fresh_clone.sh --out /tmp/rc.md                      # HEAD of this checkout
bash scripts/verify_fresh_clone.sh --ref <branch or tag> --denylist <your private deny-list> --out /tmp/rc.md
```

The script clones this repository at the given ref into a temporary directory, so nothing uncommitted or git-ignored can
leak into the result. It then repeats the CI host job (`.github/workflows/ci.yml`) step by step in a fresh virtualenv.
The interpreter is python3.10 when it is on PATH (as in CI), otherwise the newest python3 >= 3.10. When Docker or the
CDK tools are present it also runs the two optional test groups described below; `--skip-docker`, `--skip-cdk` and
`--cdk-python <python>` control them, and `--keep` retains the clone and logs. It also scans the whole git history for
12-digit account ids and checks that every path the README names in backticks exists. Healthy: a Markdown report ending
in `- steps: N PASS, 0 FAIL, M SKIP` and `- RESULT: PASS (<sha>)`, exit code 0. The optional groups report SKIP, never
FAIL, when their tools are missing. A FAIL in any step gives exit 1. Budget a few minutes; the `pip install` takes most
of that time. `--help` lists every step.

### The pieces, one by one

These are the same checks CI runs, for when one of them fails and you want to iterate. Run them from the repository root
in a virtualenv with `pip install -r requirements-dev.txt` (Python 3.10, the oldest interpreter the laptop side supports).

| command | proves | healthy output |
|---|---|---|
| `bash scripts/scrub_check.sh` | no 12-digit account id, access key, Hugging Face token, private-key header, RFC 1918 address, Amazon-internal hostname or e-mail, EC2 id or CDK bucket name in any tracked file, no string from the private `.scrub_denylist` (if present), and the git-ignored `local_folder/` (a private working folder that must never be committed) is not tracked; `--history` checks every commit and ref the same way plus author identities | `scrub CLEAN: N files`, with `(no .scrub_denylist present: names not checked)` when there is no deny-list |
| `detect-secrets scan --baseline .secrets.baseline` | no new secret-looking string outside the audited baseline; every entry in `.secrets.baseline` is audited as a false positive (`is_secret: false`). A real finding is fixed; a false positive is cleared with `detect-secrets audit .secrets.baseline` or a `# pragma: allowlist secret` comment (docs/SECURITY.md, Scanner suppressions). CI additionally fails on any unaudited entry | `git diff .secrets.baseline` empty, no entry without `is_secret` |
| `pytest` | the host tests (`pytest.ini` selects `-m "not docker and not aws"`): the config schema, the exact training command line the registry renders for ACT, every `check_dataset.py` failure mode on the synthetic fixture, preflight rows, job specs, the wait loop, report verdicts, how the job entrypoint handles a stop signal (SIGTERM), doc tables and citations, the scrub gate itself, and the solution identity (the `AWSSOLUTION/<id>/<version>` User-Agent token on a real signed request, the stack `Description` in the committed template) | all pass; `tests/test_cdk_template.py` is skipped whenever `aws_cdk` cannot be imported, which is the normal state of this venv on every Python version (`aws-cdk-lib` is installed only from `cdk/requirements.txt`, see the CDK tests below) |
| `python3 -m pipeline.policies_doc --check` | `docs/POLICIES.md` and the four marked README blocks (licence disclaimer, capability matrix, scenario table, instance fallback tables) are generated from `trainer/policies/*.yaml` and are current | `docs/POLICIES.md up to date; README blocks up to date` |
| `shopt -s globstar nullglob; for f in **/*.sh; do bash -n "$f"; done` | every shell script parses | no output |
| `bandit -q -r pipeline trainer cdk scripts inference -x tests,cdk/.venv,cdk/cdk.out,cdk/cdk.out-test -ll` | no Medium or High finding | no output, exit 0 |

### Docker lane (optional: needs a local Docker daemon and the training image)

```bash
pytest -o addopts="" -q -m docker -rfEs          # image act-sagemaker-training:dev, or set LEROBOT_TRAIN_IMAGE
```

This group of tests (the "docker lane") runs inside the training image. Every docker-marked test runs the image with
`--network none` and `HF_HUB_OFFLINE=1`; the fixture and predict tests also run as your own uid. Three modules carry the
marker: `tests/test_fixtures.py`, `tests/test_predict_docker.py` and `tests/test_eo1_pixel_budget_docker.py`. They show
what no host test can:

1. `tests/fixtures/make_mini_v3.py --container` records a real H.264 LeRobot v3.0 dataset through lerobot 0.6.1 inside
   the image, and lerobot loads it back. The host-mode fixture (zero-byte videos) loads in the image too.
2. `tests/test_predict_docker.py` runs the whole train, publish, predict chain on that fixture in one module-scoped
   container: `trainer/train_policy.sh` trains ACT for 20 CPU steps, `trainer/publish.py` publishes the newest checkpoint
   with `SUMMARY.json`, and `inference/predict.py` loads it offline and prints an action chunk of the trained shape (each
   predict case runs in its own container on the same mount). Two negative cases keep it honest. A truncated
   `model.safetensors` must make `predict.py` fail. A corrupted payload that yields an all-NaN chunk with exit 0 is a
   known gap, reported as `xfailed`; the only other accepted outcome is a refusal that names the non-finite chunk (the
   fix landing), anything else fails the test.
3. `tests/test_eo1_pixel_budget_docker.py` measures EO-1's image budget under the pinned transformers (see
   `docs/TRAPS.md`). It first downloads the Qwen2.5-VL processor files (about 12 MB, not gated) on the host, so it needs
   internet access on the host and skips without it. The container itself stays offline.

When `docker` is missing, the daemon does not answer, or the image is absent, the whole group skips with the reason and
never fails. Healthy: every docker-marked test passes except the two known gaps, which are reported as `xfailed`
(`predict.py` accepting a corrupted payload; transformers 5.5.4 ignoring `processor_kwargs`). The summary line then
reads `7 passed, N deselected, 2 xfailed`; the deselected N are the host tests, which `-m docker` leaves out. When the
host has no internet access, the EO-1 module skips instead and the line reads `5 passed, 3 skipped, N deselected,
1 xfailed`. Without `aws-cdk-lib` in the venv the line carries one more `skipped`, which is `tests/test_cdk_template.py`
(no `aws_cdk`), not a docker test.

The fixture tests take seconds. The predict module takes about a minute on a many-core host (52 s measured on 32 cores,
77 s for the whole group) and several minutes on a laptop; the build container has a 7 minute timeout. Almost all of that
time goes into the container that encodes the fixture videos and trains the 20 steps. The test sets `OMP_NUM_THREADS=8`
so the trainer does not oversubscribe the cores. Add `-s` to see the per-step timing lines.

The image is the one the stack's CodeBuild project pushes to ECR (`EcrRepoUri` output, digest in SSM
`/<project>/image_uri`). After an ECR login, `docker pull` it and `docker tag` it `act-sagemaker-training:dev`, or set
`LEROBOT_TRAIN_IMAGE` to the pulled name. `trainer/sagemaker/build_and_push.sh --region <region> [--profile <profile>]`
is the maintainer fallback and not part of this credential-free group. It builds the same Dockerfile locally, then uses
AWS credentials to log in to ECR, create the `<project>-sagemaker-training` repository if needed (the one the stack
creates, `lerobot-finetune-sagemaker-training` by default) and push the image. That is an AWS write outside the CDK
stack and about 13 GB of compressed upload; the stack's CodeBuild build is the normal path.

### CDK lane (optional: needs Python >= 3.10, Node 22 and the `cdk` CLI; no credentials)

```bash
npm install -g aws-cdk
pip install -r requirements-dev.txt -r cdk/requirements.txt      # into the Quick start virtualenv (Python >= 3.10)
python -m pytest tests/test_cdk_template.py -m "not docker and not aws"
```

The test synthesizes the stack itself. The fixture pins `AWS_REGION` and `AWS_DEFAULT_REGION` to `us-east-1` and sets
`HF_TOKEN_PROMPT=0`; no account is contacted, no Hugging Face call is made and no token prompt can appear. It then
checks, one test each:

- the resource set is exactly what the README promises: one ECR repository, one bucket with its policy, four IAM roles
  and five policies, five SSM parameters, two log groups (image build, helper Lambdas), one CodeBuild project, two helper
  Lambdas, and four custom resources (the build kickoff, the S3 auto-delete, the SSM image-parameter cleanup and the Hub
  token secret cleanup);
- the ten stack outputs;
- bucket and repository `Delete` by default (`EmptyOnDelete`, S3 auto-delete armed by its tag), the two cleanups' delete
  calls, both helper Lambdas logging to a stack-managed log group, plus the bucket's public-access block;
- with `-c retainData=true`: bucket, bucket policy and repository `Retain`, no auto-delete resource, the secret cleanup
  kept with `Retain`;
- the execution role may put and delete only under `sagemaker/jobs/*` and `sagemaker/hub/*`;
- no managed policy on the SageMaker execution role, and a trust policy that admits SageMaker only for this account's resources (`aws:SourceAccount`, `aws:SourceArn`);
- CodeBuild uses the `-f` Dockerfile path in privileged mode;
- cdk-nag reports zero unsuppressed findings;
- the committed plain CloudFormation export under `cdk/template/` matches a fresh synth, including the pins in
  `cdk/requirements.txt`.

Healthy: all tests pass. A drift failure means the stack changed without re-running the export. The test prints the
regeneration command; [cdk/template/README.md](../cdk/template/README.md) ("Regenerating") explains it and the pin-bump
procedure.

## 2. The AWS quick start, annotated

Follow the README "Quick start" as written. This section says what each step printed on a fresh clone of this branch,
in the region the stack was deployed in, with the shipped `project_config.json` (public dataset `lerobot/svla_so101_pickplace`, ACT,
`ml.g6e.4xlarge`), against a stack that already existed. Prompts: `finetune.sh` prints one plan table and asks for a
single confirmation before any AWS write (uploads, base-weight staging, smoke and training jobs). `--plan` stops after
the table. A non-interactive shell needs `--yes`, otherwise the run stops with exit 2 at that question.

| step | healthy result | time | cost |
|---|---|---|---|
| `pip install -r requirements.txt -r cdk/requirements.txt` in a fresh Python 3.10 venv | installs boto3, PyYAML, pyarrow, numpy, jsonschema, huggingface_hub, aws-cdk-lib and cdk-nag | 1 min | 0 |
| `cdk bootstrap` then `cdk deploy` in `cdk/` | first deploy: stack created and the image build started on CodeBuild (~15 min in the background, `KickoffBuildId` output). A later deploy of a one-resource change: `UPDATE_COMPLETE` in 47 s and no new image build (the build starts again only when `trainer/` or `image.tag` changes) | a few minutes for the first deploy (47 s measured for a one-resource update) + 15 min build | ~$0.50 once |
| `./finetune.sh --dry-run` before the build has published `/<project>/image_uri` | `dataset:` runs first (download and check), then `image: the deploy-time build is still running (started <hh:mm> UTC, about 15 min in total); waiting` with the CodeBuild console link, one line per build phase, then the run continues as below. Ctrl-C leaves the build running; the next run picks it up again. A failed build exits 4 with the console link | the rest of the build | 0 |
| `./finetune.sh --dry-run` afterwards | a `config:` line; `dataset:` downloads the 86 MB dataset from the Hub on the first run and checks it: `OK: 0 fail(s), 4 warning(s)` (two cameras are 4:3, two features lack `q01`/`q99` stats) and `split: 45 training episodes, held-out [0, 10, 20, 30, 40]`; a `stack:` line; `preflight:` with five OK rows (`licence`, `credentials`, `region`, `image`, `quota <name> = <value>`); the `plan:` table; `stage:` with either `will upload N files, M MB -> s3://...` or `already staged (fingerprint ...)`; then the smoke job spec and the training job spec as JSON, each followed by `(dry run: not submitted)` | 19 s | 0 |
| `./finetune.sh --yes`, smoke stage | `[act] smoke: 200 steps on ml.g6e.4xlarge (~<$1)`, `submitted <job> <console url>`, one line per SageMaker `SecondaryStatus` transition, `<job>: Completed, billable 436 s`, `smoke gate PASS: step rate <rate> steps/s, billable 436 s` (about 16 steps/s on this instance) | 8 min | < $1 |
| `./finetune.sh --yes`, train stage | `[act] train: 100000 steps on ml.g6e.4xlarge, ~1.7 h at <rate> steps/s, runtime cap ~5.6 h, volume 100 GB`, the hyperparameters, `submitted <job> <console url>`, `reattach any time with: ./finetune.sh --train-job <job>`, then every 30 s the newest `step:... loss:...` record from the job's `train.log`, then `Completed, billable <s> s`. The cap is 15 min start-up + 20 min evaluation + ~1.7 h x `compute.runtime_multiplier` (3.0 in the shipped `project_config.json`; the 1.5 of `examples/project_config.vla.json` gives 3.1 h) | 1 h 54 min measured (6822 s billable) | ~$10 |
| report stage (automatic) | a table with `this run` and the two `ACT public reference` rows (the 2026-09-10 on-demand run and the 2026-09-14 managed Spot run), `verdict: n/a` with `no reference band for this dataset; numbers reported without a verdict`, and `artefacts: runs/<name>/model (...), runs/<name>/MANIFEST.json` | seconds | 0 |

Exit codes are listed in [docs/ARCHITECTURE.md](ARCHITECTURE.md) (the stage table). Two details matter here: a
confirmation prompt without a terminal exits 2, and a stack missing in this region or any failed AWS call exits 4.
Ctrl-C never stops a job. `./finetune.sh --train-job <job-name>` reattaches, waits and runs the report stage;
`python3 -m pipeline.wait <job-name> --region <region>` only waits; `python3 -m pipeline.report --run
runs/<dataset>-act.json` prints the report of a finished run again, and `--compare runs/*.json` tabulates several.

Where things land (the bucket name is the `BucketName` stack output; the dry-run specs show the full URIs):

| location | content |
|---|---|
| `s3://<bucket>/sagemaker/datasets/<name>/` | the dataset, mirrored with `aws s3 sync --delete`; `sagemaker/datasets/_staged/<name>.json` records its fingerprint |
| `s3://<bucket>/sagemaker/jobs/smoke/<job>/checkpoints/act/` | the smoke job's `train/checkpoints/<step>/`, `train.log`, `split.json`, `smoke_checks.json`, `SUMMARY.json`; a bucket lifecycle rule expires this prefix after 14 days |
| `s3://<bucket>/sagemaker/jobs/train/<job>/checkpoints/act/` | the same for the training job, synced while it runs. One checkpoint per `save_freq` = 5000 steps, but only the newest `compute.keep_checkpoints` = 3 complete ones are kept and the S3 mirror follows the deletes (`0` keeps all 20). This is what `--resume-from <job>` restores |
| `s3://<bucket>/sagemaker/jobs/train/<job>/output/` | `model.tar.gz` (published checkpoint, evaluation, summary, licence notice) and `output.tar.gz` (logs) |
| `runs/<dataset>-act.json` | the laptop's run state: jobs, fingerprints, smoke summary, final summary, verdict (git-ignored) |
| `runs/<dataset>-act/model/` | `pretrained_model/` (loadable with stock lerobot), `eval/eval_last.json` + `eval_last_preds.npz`, `SUMMARY.json`, `LICENCE_NOTICE.md` |
| `runs/<dataset>-act/output/` | `train.log`, `losscurve.txt`, `split.json`, `exitcode` |
| `runs/<dataset>-act/MANIFEST.json` | job name, status, instance type, billable seconds, Spot flag, `network_isolation`, image digest, hyperparameters, dataset URI, fingerprint, object count and bytes, split, checkpoint step, model sha256, versions, licence status, `repo_git_sha`, `model_artifact` |

`SUMMARY.json` (written in the job by `trainer/summary.py`) carries `policy`, `status` (`completed` or `interrupted`),
`checkpoint_step`, `model_sha256`, `split`, `evals`, `step_rate`, `dataset_fingerprint`, `image_digest`, `licence_status`,
`deliverable`, `versions`, `reload_check` (`ok`, `max_abs_diff_vs_eval`, `pred_shape`) and `train_exitcode`. `evals`
holds, per evaluated checkpoint: `layout`, `joints_mae`, `joints_hold`, `grippers_mae`, `grippers_hold`,
`gripper_timing`, `beats_hold`, `episodes`, `stride`, `chunk` and `n_samples`. `step_rate` is the median training speed
in steps per second, measured from the log; it can differ by about 2 % from the steady-state rate that `docs/RESULTS.md`
quotes, and it is null when it could not be measured. The smoke job's copy adds `smoke` (`weights_moved_fraction`,
`bf16_trainable_params`).

Compare your numbers with the ACT rows of [docs/RESULTS.md](RESULTS.md) ("Full fine-tunes on the public dataset"; same
config, same image family). Expect about 6800 s billable (~$10), `checkpoint_step` 100000 and `layout` `generic_6`.
Expect `joints_mae` around 4.3 in the dataset's units (degrees) against a hold baseline of 10.15, so about 0.42 x hold
(the metric is defined in docs/RESULTS.md), `beats_hold` 6/6 and `chunk` 30 (1.0 s at 30 fps). Expect `reload_check.ok`
true with a diff of 0.0 on the training GPU. `pipeline/reference_results.json` lists the two reference runs for this dataset (4.305 and
0.424 x hold on-demand, 4.274 and 0.421 x hold on managed Spot) without an acceptance band, so the verdict stays `n/a`.
A healthy repeat lands near those numbers; the fresh-clone repeat of 2026-09-11 landed within 1 % on joints MAE. Exact
agreement is not expected (the two reference runs on the private dataset differ by about 0.0005 in MAE; cuDNN is not
deterministic). Read the job log when the ratio is near 1.0 x hold, `beats_hold` is far below 6/6, or `reload_check.ok`
is false.

## 3. Optional deeper checks

### Load the checkpoint and predict, offline

`inference/predict.py` is the minimal deployment example. It loads `runs/<run>/model/pretrained_model/` with stock
lerobot, takes one observation from a dataset and prints the predicted action chunk. It needs lerobot 0.6.1, which the
laptop does not have, so run it inside the training image with the same flags the docker tests use:

```bash
docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=/tmp/hf -e HF_HUB_OFFLINE=1 \
  -v "$PWD:/work:ro" -w /work --entrypoint python act-sagemaker-training:dev \
  inference/predict.py --model-dir runs/<dataset>-act/model --dataset-root datasets/<dataset> --episode 0 --frame 0
```

Healthy, for the public dataset: `policy act: 30 future actions x 6 dims for episode 0 frame 0`, five rows of joint
targets in degrees, `dims: shoulder_pan.pos, ..., gripper.pos`, and no network access attempted. This runs on the dataset
at its recorded resolution. The job trained and evaluated on a copy downscaled to height 360, so the numbers are not the
evaluation's. For a bit-exact comparison, make that copy first, inside the image and with a writable mount. The script
needs ffmpeg and ffprobe (in the image, not on the laptop) and writes `data/`, `meta/` and `videos/` into the
destination. No Hugging Face environment is needed for this step:

```bash
docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD:/work" -w /work --entrypoint bash act-sagemaker-training:dev \
  trainer/prepare_dataset_360.sh datasets/<dataset> datasets/<dataset>-360
```

Then run the check the job ran, `trainer/reload_check.py`, against that copy (read-only mount again):

```bash
docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=/tmp/hf -e HF_HUB_OFFLINE=1 \
  -v "$PWD:/work:ro" -w /work --entrypoint python act-sagemaker-training:dev \
  trainer/reload_check.py --model-dir runs/<dataset>-act/model --dataset-root datasets/<dataset>-360 --episodes 0 10 20 30 40 --policy act --tol 0.05
```

It prints a JSON with `ok`, `pred_shape` (`[30, 6]` here) and `max_abs_diff_vs_eval`, the largest difference between
this prediction and the one the in-job evaluation stored in `eval/eval_last_preds.npz`. The in-job check on the training
GPU reported 0.0 (the reload check that `docs/RESULTS.md` records as passed). A CPU re-run of the same checkpoint differs
by device rounding, on the order of 1e-2 on joint angles of about 100 degrees (relative 2e-4). So pass `--tol 0.05`
instead of the in-job default of 1e-4 and read the printed `max_abs_diff_vs_eval`. Anything larger than a fraction of a
degree means the checkpoint or the dataset copy is not the one that was evaluated (running against the raw, un-downscaled
dataset gives several degrees).

The sampling policies (SmolVLA, X-VLA, pi0, pi0.5) have flow-matching heads that draw fresh noise on every call, and
their evaluation averages 4 draws. The job runs the check for them with `--stochastic`, and so must you. Then
`max_abs_diff_vs_eval` compares one draw with the stored 4-sample mean and is for information only. `ok` means the
checkpoint loads offline and predicts a chunk of the right shape (`[50, 6]` for SmolVLA, pi0 and pi0.5, `[24, 6]` for
X-VLA), not that a number was reproduced. In the recorded runs of these policies on the public dataset, that difference
was between about 1 and 18 degrees: 13.4 for SmolVLA and 1.1 for X-VLA inside the job, 17.9 and 17.8 for CPU re-draws of
the same checkpoints in the release image. None of that is an error. A load error, a wrong shape or non-finite values
are.

### Your own dataset before it costs anything

[docs/RECORDER_CHECK.md](RECORDER_CHECK.md) explains how to check a recording on the robot PC before you copy it. On the
laptop, `./finetune.sh --stop-after dataset` runs the same check with the flags the pipeline adds and stops before any
AWS call (no stack needed). Then follow README Quick start step 4 to point `project_config.json` at your folder.

### Several policies in one run

`examples/project_config.vla.json` enables SmolVLA, X-VLA, pi0 and pi0.5 on the public dataset. The quickest look is
`./finetune.sh --config examples/project_config.vla.json --dry-run`; [examples/README.md](../examples/README.md) says what
the file runs, what it needs and the two ways to use it. The file already sets `accept_licence` for X-VLA, pi0 and
pi0.5. pi0 and pi0.5 also need a Hugging Face token, which `./finetune.sh` asks for (hidden input) at their
`base weights:` step; a non-interactive run (`CI=1`) exits 4 there instead (user steps: README, "Gated base weights and
the Hugging Face token"; mechanics: `docs/ARCHITECTURE.md`, stages of `./finetune.sh`).

The dry run prints, in this order:

1. `dataset:` once per distinct dataset; a per-policy `dataset` override prints
   `dataset (<policies>): <name> (per-policy override)`.
2. The `plan:` table.
3. `stage:` once per distinct dataset.
4. For each enabled policy, in config order, `[<policy>] base weights:` when the registry entry lists `hub_artifacts`.
   It says either `base weights already staged: s3://.../sagemaker/hub/<policy>/` or
   `will stage N artefact(s) for <policy> with a ml.m5.4xlarge job (network on)`. The second form is followed by one line
   per artefact (a file the staging job downloads from the Hub), with `[gated]` where a token is needed, and by a
   `stale staging` line with the object count and bytes of an earlier staging that a real run would delete first. The
   staging job (`pipeline/hub.py`) is an
   on-demand `ml.m5.4xlarge` with a 2 h cap, a 200 GB volume and network isolation off.
5. `[<policy>] smoke: 200 steps on <instance>`, or `smoke: reusing <job>` when an identical smoke already passed.
6. `[<policy>] train: <steps> steps on <instance>`.

Each job spec is followed by `(dry run: not submitted)`. Preflight adds a `quota` row for
`ml.m5.4xlarge for training job usage` when any enabled policy stages weights. A real run with several enabled policies
submits them in parallel after the single confirmation, one child process and one job per policy. The instance quota
must therefore cover them all at once; `--sequential` runs them in order in this process instead. See the README,
"Fine-tune other policies". Within a policy, staging, smoke, training and report run in order.

Per policy, check three files. `runs/<run>-<policy>/model/SUMMARY.json`: `policy` is the registry key, `status` is
`completed`, `licence_status` is `accepted`. `runs/<run>-<policy>/model/LICENCE_NOTICE.md`: the entry's licence class,
tier and deliverable flag, and an `Instrument:` line with the URL the registry entry cites.
`runs/<run>-<policy>/MANIFEST.json`: its `hyperparameters` carry `accept_licence` `"1"` for `xvla`, `pi0` and `pi05`
(SmolVLA sets none, so its MANIFEST has no such key) and, when set, `approval_ref`. `<run>` is the dataset name unless `--name` says otherwise. The recorded runs
of this configuration and their numbers are in `docs/RESULTS.md`; the status per policy is in `docs/POLICIES.md`;
neither is restated here.

### The published image after a trainer change (maintainers)

`bash scripts/smoke_aws.sh --region <region>` submits a 20-step CPU training job (`ml.c5.4xlarge`, about $0.30, 12 to
18 min) on the image the stack published and the staged dataset. It checks only that the job completes and publishes a
finite joints MAE; with `--expect-joints-mae 17.3817` it also fails unless the MAE is the value every validated image has
produced on the public dataset (the ACT fingerprint, see `python3 -m pipeline.smoke_aws --help`). It is the per-change gate for anything under `trainer/`; `docs/RESULTS.md` logs each one.

### Teardown

Follow the README "Teardown" section ([README.md](../README.md); both paths, including `scripts/destroy.sh`) and
[docs/SECURITY.md](SECURITY.md) "Costs that keep running". Nothing differs for a test account.
