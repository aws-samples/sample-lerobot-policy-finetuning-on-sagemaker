#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""CDK app: one stack with everything a fresh account needs to run ./finetune.sh.

Reads the repo-root project_config.json (project name, image.tag, secrets.hf_token) so the stack and
the pipeline can never disagree; CDK context overrides exist for CI and for a second deployment:

  cdk deploy                                    # region from the ambient chain (AWS_REGION, then the profile)
  cdk deploy -c region=eu-central-1             # override the region for this deploy
  cdk deploy -c imageBuild=skip                 # do not start an image build at deploy time
  cdk deploy -c buildCompute=XLARGE             # LARGE (default) | XLARGE | 2XLARGE for the CodeBuild image factory
  cdk deploy -c bucket_name=my-bucket           # name the bucket instead of letting CDK generate one
  cdk deploy -c retainData=true                 # keep the bucket, the images and the Hub token secret after `cdk destroy`
                                                # (default false: `cdk destroy` removes everything the deploy created)
  cdk deploy -c hfTokenPrompt=true              # ask for a Hugging Face token now and store it in Secrets Manager
                                                # (default false: finetune.sh asks when a gated policy needs it)
  cdk synth  -c plainTemplate=1                 # plain CloudFormation export: no CDK bootstrap rule, region-agnostic,
                                                # assets bucket as a template parameter (see cdk/template/README.md;
                                                # regenerate the committed copy with cdk/export_template.py)

Export the region to BOTH AWS_REGION and AWS_DEFAULT_REGION when you rely on the ambient chain: the CDK CLI reads the
first, boto3 (the pipeline) the second, and a mismatch deploys the stack in one region and trains in another.
"""
import os
import sys
from pathlib import Path

import aws_cdk as cdk
from cdk_nag import AwsSolutionsChecks

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import load_project_config  # noqa: E402
from pipeline.solution import get_client, stack_description  # noqa: E402
from lerobot_finetune.stack import LeRobotFinetuneStack  # noqa: E402

cfg = load_project_config()
app = cdk.App()

# Every taggable resource carries the project name; add your account's own tags here the same way, or at deploy
# time with `cdk deploy --tags key=value` (stack-level).
cdk.Tags.of(app).add("project", cfg["project"])

project = app.node.try_get_context("project") or cfg["project"]
region = app.node.try_get_context("region") or None
image_tag = app.node.try_get_context("imageTag") or cfg["image"]["tag"]
build_compute = (app.node.try_get_context("buildCompute") or "LARGE").upper()
image_build = (app.node.try_get_context("imageBuild") or "kickoff").lower()
bucket_name = app.node.try_get_context("bucket_name")
hf_token_secret = app.node.try_get_context("hfTokenSecret") or cfg.get("secrets", {}).get("hf_token") or f"{project}/hf_token"
_retain_ctx = app.node.try_get_context("retainData")
retain_data = (str(_retain_ctx).lower() not in ("0", "false", "no")) if _retain_ctx is not None else False
# plainTemplate=1: synth a template deployable with `aws cloudformation deploy` and no CDK toolkit. The default
# synthesizer still names the two file assets by content hash, but (a) no CheckBootstrapVersion rule / BootstrapVersion
# SSM parameter, (b) the assets bucket is the template parameter AssetsBucketName instead of the hard-coded bootstrap
# bucket, and (c) no region is baked in. `cdk deploy` never sets this; it changes nothing for the default path.
PLAIN_TEMPLATE_PARAM = "AssetsBucketName"
plain_template = str(app.node.try_get_context("plainTemplate") or "").lower() in ("1", "true", "yes")
# The execution role may read that one secret. With -c hfTokenPrompt=true on an interactive `cdk deploy` (a terminal, not
# CI, not plainTemplate) the operator is prompted once for a Hugging Face token when the secret is missing or rejected by
# the Hub; it is validated and stored in Secrets Manager, never written to disk here. Off by default: the CDK CLI runs this
# app for `cdk bootstrap`, `cdk synth` and `cdk diff` too, and a prompt there would fire before the bootstrap stack exists
# and on every credential-free test synth. finetune.sh asks for the token when a policy with gated base weights needs it.
# HF_TOKEN_PROMPT=0 or CI=1 in the environment disables the prompt even when the context flag is set.
if str(app.node.try_get_context("hfTokenPrompt") or "false").lower() in ("1", "true", "yes") and not plain_template:
    try:
        import boto3
        from pipeline import hf_token as _hf
        if _hf.interactive():
            # CDK_DEFAULT_REGION is what the CDK CLI resolved (from --profile or the ambient chain) and will deploy to;
            # boto3 ignores it, so pin it explicitly or the secret could land in another region than the stack
            _prompt_region = region or os.environ.get("CDK_DEFAULT_REGION")
            _sess = boto3.Session(region_name=_prompt_region) if _prompt_region else boto3.Session()
            # codeql[py/clear-text-logging-sensitive-data] the secret's name and the Hub user, never the token (see hf_token._emit)
            _hf.ensure(get_client("secretsmanager", session=_sess), hf_token_secret, log=lambda m: print(m, file=sys.stderr), required=True)
    except (Exception, KeyboardInterrupt) as _e:   # a failed, skipped or interrupted prompt never blocks synth; finetune.sh asks again
        print(f"NOTE: Hugging Face token not stored now ({type(_e).__name__}: {_e}); finetune.sh asks when a policy needs it", file=sys.stderr)

synthesizer = None
if plain_template:
    synthesizer = cdk.DefaultStackSynthesizer(generate_bootstrap_version_rule=False,
                                              file_assets_bucket_name=f"${{{PLAIN_TEMPLATE_PARAM}}}")
    region = None

stack = LeRobotFinetuneStack(
    app, f"{project}-infra",
    project=project, image_tag=image_tag, build_compute=build_compute,
    kickoff_build=(image_build != "skip"), bucket_name=bucket_name, hf_token_secret_name=hf_token_secret, retain_data=retain_data,
    env=cdk.Environment(account=app.node.try_get_context("account") or None, region=region),
    synthesizer=synthesizer,
    # "(<id>) - <name>. Version <version>" from the solution block of project_config.json: deployment counting matches the
    # id in parentheses and nothing else about the stack. One stack per install, so no component label and no id suffix;
    # a second stack added later must pass id_suffix= or one install counts twice (see pipeline/solution.py). Deleting the
    # block makes this raise: a stack that deploys without the id is counted as zero installs, silently, forever.
    description=stack_description(),
)
if plain_template:
    cdk.CfnParameter(stack, PLAIN_TEMPLATE_PARAM, type="String", min_length=3, max_length=63,
                     allowed_pattern="^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$",  # general-purpose S3 bucket naming rules
                     constraint_description="an S3 bucket name: 3-63 lowercase letters, digits, dots or hyphens, "
                                            "starting and ending with a letter or digit",
                     description="S3 bucket (same region as the stack) holding the asset zips listed in cdk/template/assets.json")
# Every synth is a compliance scan; an unsuppressed AwsSolutions finding fails it. Deviations are suppressed on the
# resource that carries them, with the reason inline (see stack.py).
cdk.Aspects.of(app).add(AwsSolutionsChecks())
app.synth()
