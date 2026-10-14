# Example configurations

Each file here is a complete `project_config.json` for one recorded run. There are two ways to use one:

- Run it in place: `./finetune.sh --config examples/<file>`. Add `--plan` first to see the plan and the estimated cost
  without submitting anything.
- Copy it over the repo-root `project_config.json` (or merge its `policies` block into yours) and run `./finetune.sh`.

The CDK stack is always deployed from the repo-root `project_config.json`, whichever way you run an example. No config
carries a region: export `AWS_DEFAULT_REGION` (and `AWS_REGION`) as the region where your stack is deployed, or pass
`--region <region>`. Without a region the run stops with "region is required" (exit code 2); "stack not found" (exit
code 4) means the region is set but no stack is deployed there. The pipeline checks every config against
`pipeline/project_config.schema.json`.

| file | what it runs | what you need in your account | recorded run ([docs/RESULTS.md](../docs/RESULTS.md)) |
|---|---|---|---|
| `project_config.vla.json` | SmolVLA (20k steps), X-VLA (30k), pi0 (20k) and pi0.5 (20k) on `lerobot/svla_so101_pickplace`. Per policy: a CPU staging job on `ml.m5.4xlarge` that downloads the base weights into your bucket, then a 200-step smoke test and the training job, both on `ml.g6e.4xlarge` on-demand | Quota: `ml.g6e.4xlarge for training job usage` of 4 (one per policy running at once, or 1 with `--sequential`) and `ml.m5.4xlarge for training job usage` of at least 1 for the staging jobs. Licence: the file already sets `accept_licence: true` for X-VLA, pi0 and pi0.5. Token: a Hugging Face token in Secrets Manager (`lerobot-finetune/hf_token`) whose account has accepted Google's terms for the `google/paligemma-3b-pt-224` tokenizer that pi0 and pi0.5 download | 2026-09-11: full fine-tunes of all four on `ml.g6e.4xlarge` |
| `project_config.80gb.json` | pi0_fast, EO-1 and FastWAM, three of the four offered policies that need an 80 GB GPU (fastwam_lora is the fourth), on `hjkso1406/so101-4tasks-100eps`. Per policy: the same staging job, then a 200-step smoke test and the training job on `ml.p5.4xlarge` Spot | Quota: `ml.p5.4xlarge for spot training job usage` (the file sets `compute.spot: true`; with Spot off, the on-demand name) and `ml.m5.4xlarge for training job usage` for the staging jobs. pi0_fast and EO-1 also accept `ml.g7e.4xlarge` (untested). Licence: the file already sets `accept_licence: true` for all three. `approval_ref` is `null` for EO-1 (research-only licence) and FastWAM (licence unknown): write your own reference there (who read the terms, and where that is recorded), or `finetune.sh` refuses both before it submits anything ([docs/POLICIES.md](../docs/POLICIES.md)). Token: a Hugging Face token as above (pi0_fast downloads the same tokenizer) | 2026-09-15: 200-step smoke tests only. They ran on EC2 H100 hosts with the same container, because no 80 GB SageMaker instance was available at the time. This file has not yet been run on SageMaker |

With several policies enabled, `finetune.sh` prints one plan, asks once, then runs one child process per policy
(`--sequential` runs them one after another in one process). The staging job is skipped when the weights are already in your
bucket, and the smoke test when an identical one already passed. Both files evaluate on a subset of the held-out frames
(every 10th frame in the VLA file, every 60th in the 80 GB file), which keeps the evaluation to minutes. When the token
secret is missing, `finetune.sh` asks for the token once and checks only that the Hub accepts it. Whether the account has
accepted Google's terms shows up when the staging job downloads the tokenizer. How the parallel run, the licence gate and the
token prompt work is described in [README.md, "Fine-tune other policies"](../README.md#fine-tune-other-policies). The quota
names and how to raise a quota are in [README.md, "Quotas"](../README.md#quotas). Adding `--instance-type auto` to either
command makes the run pick, per policy, the first type of the registry's `hardware.instance_types` list that has quota in
your region, so the GPU quota named above becomes a preference for the VLA file and for pi0_fast and EO-1 (which fall back
to `ml.g7e.4xlarge`); FastWAM lists only multi-GPU alternatives, which `auto` skips without `--allow-multigpu`, so its
`ml.p5.4xlarge` quota stays required. The `ml.m5.4xlarge` quota for the staging jobs, the licence fields and the token are
required either way.
