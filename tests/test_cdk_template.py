# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""The stack must contain exactly the resources the README promises and pass cdk-nag with zero unsuppressed findings.

Runs where aws-cdk-lib is importable (the cdk/.venv on Python >= 3.10, or the CI cdk job); a host suite without
aws-cdk-lib skips it.
"""
import csv
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

aws_cdk = pytest.importorskip("aws_cdk")
REPO = Path(__file__).resolve().parent.parent
CDK = REPO / "cdk"
EXPECTED_TYPES = {   # the shipped default: retainData unset, so `cdk destroy` removes everything
    "AWS::ECR::Repository": 1, "AWS::S3::Bucket": 1, "AWS::S3::BucketPolicy": 1,
    "AWS::IAM::Role": 4,          # SageMaker execution, CodeBuild, the custom-resource Lambda, the S3 auto-delete provider
    "AWS::IAM::Policy": 5,        # execution role, CodeBuild, kickoff, SSM-parameter cleanup, secret cleanup
    "AWS::SSM::Parameter": 5,
    "AWS::Logs::LogGroup": 2,     # image build logs, helper-Lambda logs (both deleted with the stack)
    "AWS::CodeBuild::Project": 1,
    "AWS::Lambda::Function": 2,   # the AwsCustomResource singleton, the S3 auto-delete provider
    "Custom::ImageBuildKickoff": 1, "Custom::ImageParamsCleanup": 1, "Custom::HubTokenSecretCleanup": 1,
    "Custom::S3AutoDeleteObjects": 1,
}
# -c retainData=true: no auto-delete provider; the bucket, the repository and the secret outlive the stack
EXPECTED_TYPES_RETAIN = dict(EXPECTED_TYPES, **{"AWS::IAM::Role": 3, "AWS::Lambda::Function": 1})
EXPECTED_TYPES_RETAIN.pop("Custom::S3AutoDeleteObjects")
EXPECTED_OUTPUTS = {"Project", "Region", "Account", "BucketName", "S3Root", "EcrRepoUri", "SageMakerRoleArn",
                    "ImageBuildProject", "SsmParameterPrefix", "KickoffBuildId"}
sys.path.insert(0, str(REPO))
from pipeline.config import load_project_config  # noqa: E402
from pipeline import solution  # noqa: E402
_CFG = load_project_config()
PROJECT = _CFG["project"]
HF_SECRET = (_CFG.get("secrets") or {}).get("hf_token") or f"{PROJECT}/hf_token"


# Every synth here is credential-free: the region is pinned so the CDK CLI never resolves one, and HF_TOKEN_PROMPT=0 makes
# app.py's optional Hugging Face token prompt (-c hfTokenPrompt=true) a no-op even if someone sets that flag, so no test
# reads a secret, calls huggingface.co or blocks on a hidden prompt (pipeline/hf_token.py `interactive()`).
SYNTH_ENV = dict(os.environ, AWS_REGION="us-east-1", AWS_DEFAULT_REGION="us-east-1", HF_TOKEN_PROMPT="0")


@pytest.fixture(scope="module")
def synth():
    env = SYNTH_ENV
    out = CDK / "cdk.out-test"
    r = subprocess.run(["cdk", "synth", "--quiet", "-o", str(out), "--app", f"{sys.executable} app.py"],
                       cwd=CDK, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    (template,) = glob.glob(str(out / "*infra.template.json"))
    return json.load(open(template)), out


def test_resource_set_is_exactly_what_the_readme_promises(synth):
    template, _ = synth
    counts = {}
    for res in template["Resources"].values():
        counts[res["Type"]] = counts.get(res["Type"], 0) + 1
    counts.pop("AWS::CDK::Metadata", None)
    assert counts == EXPECTED_TYPES


def test_description_carries_the_solution_id(synth):
    """Deployment counting matches the solution id in parentheses in the stack Description and nothing else; one stack
    per install, so the bare id, no suffix. Read from the synthesized template, not the source."""
    template, _ = synth
    desc = template["Description"]
    assert desc == solution.stack_description(), desc
    assert desc.startswith(f"({solution.SOLUTION_ID}) - ") and desc.endswith(f". Version {solution.SOLUTION_VERSION}"), desc
    assert len(desc.encode("utf-8")) <= 1024


def test_outputs(synth):
    template, _ = synth
    assert set(template["Outputs"]) == EXPECTED_OUTPUTS


def test_bucket_hardening_and_retention(synth):
    template, _ = synth
    bucket = next(r for r in template["Resources"].values() if r["Type"] == "AWS::S3::Bucket")
    assert bucket["DeletionPolicy"] == "Delete"          # the default removes the data with the stack
    pab = bucket["Properties"]["PublicAccessBlockConfiguration"]
    assert all(pab[k] for k in ("BlockPublicAcls", "BlockPublicPolicy", "IgnorePublicAcls", "RestrictPublicBuckets"))
    repo = next(r for r in template["Resources"].values() if r["Type"] == "AWS::ECR::Repository")
    assert repo["DeletionPolicy"] == "Delete" and repo["Properties"].get("EmptyOnDelete") is True
    assert repo["Properties"]["ImageScanningConfiguration"]["ScanOnPush"] is True


def test_cdk_destroy_leaves_nothing_of_its_own_behind(synth):
    """The default stack removes what CloudFormation would otherwise leave: bucket contents (auto-delete), images
    (EmptyOnDelete), the two SSM parameters the build writes, the Hub token secret, and both helper Lambdas' logs."""
    template, _ = synth
    res = template["Resources"]
    assert any(v["Type"] == "Custom::S3AutoDeleteObjects" for v in res.values())
    deletes = {v["Type"]: json.loads(v["Properties"]["Delete"]) for v in res.values()
               if v["Type"] in ("Custom::ImageParamsCleanup", "Custom::HubTokenSecretCleanup")}
    assert deletes["Custom::ImageParamsCleanup"]["action"] == "deleteParameters"
    assert deletes["Custom::ImageParamsCleanup"]["parameters"]["Names"] == [f"/{PROJECT}/image_uri", f"/{PROJECT}/image_src_hash"]
    assert deletes["Custom::HubTokenSecretCleanup"]["parameters"] == {"SecretId": HF_SECRET, "ForceDeleteWithoutRecovery": True}
    bucket = next(v for v in res.values() if v["Type"] == "AWS::S3::Bucket")
    assert {"Key": "aws-cdk:auto-delete-objects", "Value": "true"} in bucket["Properties"]["Tags"], "auto-delete is armed by this tag"
    kickoff = next(v for v in res.values() if v["Type"] == "Custom::ImageBuildKickoff")
    assert "Delete" not in kickoff["Properties"]
    for v in res.values():   # the cleanups do nothing at create/update time
        if v["Type"] in ("Custom::ImageParamsCleanup", "Custom::HubTokenSecretCleanup"):
            assert "Create" not in v["Properties"] and "Update" not in v["Properties"]
    log_groups = {k for k, v in res.items() if v["Type"] == "AWS::Logs::LogGroup"}
    assert all(v["DeletionPolicy"] == "Delete" for k, v in res.items() if k in log_groups)
    functions = [v for v in res.values() if v["Type"] == "AWS::Lambda::Function"]
    assert len(functions) == 2 and all(v["Properties"]["LoggingConfig"]["LogGroup"]["Ref"] in log_groups for v in functions), \
        "every helper Lambda must log to a stack-managed group, or `cdk destroy` leaves /aws/lambda/<fn> behind"


def test_sagemaker_role_writes_only_under_jobs_and_hub(synth):
    """The execution role may put/delete only under sagemaker/jobs/* and sagemaker/hub/*; datasets and build prefixes are
    read-only for the job identity (docs/THREAT_MODEL.md T-28)."""
    template, _ = synth
    pol = next(v for k, v in template["Resources"].items()
               if v["Type"] == "AWS::IAM::Policy" and k.startswith("SageMakerExecutionRoleDefaultPolicy"))
    for st in pol["Properties"]["PolicyDocument"]["Statement"]:
        actions = st["Action"] if isinstance(st["Action"], list) else [st["Action"]]
        if any(a.startswith(("s3:PutObject", "s3:DeleteObject", "s3:Abort")) for a in actions):
            resources = st["Resource"] if isinstance(st["Resource"], list) else [st["Resource"]]
            for r in resources:
                suffix = r["Fn::Join"][1][-1] if isinstance(r, dict) and "Fn::Join" in r else str(r)
                assert suffix in ("/sagemaker/jobs/*", "/sagemaker/hub/*"), (actions, suffix)


def test_sagemaker_role_has_no_managed_policies(synth):
    template, _ = synth
    roles = [r for r in template["Resources"].values() if r["Type"] == "AWS::IAM::Role"]
    sm = [r for r in roles if r["Properties"]["AssumeRolePolicyDocument"]["Statement"][0]["Principal"]["Service"] == "sagemaker.amazonaws.com"]
    assert len(sm) == 1 and "ManagedPolicyArns" not in sm[0]["Properties"]


def test_sagemaker_role_trusts_only_this_accounts_sagemaker_resources(synth):
    """Confused-deputy guard (docs/THREAT_MODEL.md T-32): the service principal may assume the role only for a SageMaker
    resource of this account in this region (aws:SourceAccount + aws:SourceArn). Account, region and partition stay
    pseudo-parameter references, so the plain export remains account- and region-agnostic."""
    template, _ = synth
    roles = [r for r in template["Resources"].values() if r["Type"] == "AWS::IAM::Role"]
    (sm,) = [r for r in roles if r["Properties"]["AssumeRolePolicyDocument"]["Statement"][0]["Principal"]["Service"] == "sagemaker.amazonaws.com"]
    (statement,) = sm["Properties"]["AssumeRolePolicyDocument"]["Statement"]
    assert statement["Action"] == "sts:AssumeRole" and statement["Effect"] == "Allow"
    cond = statement["Condition"]
    assert cond["StringEquals"]["aws:SourceAccount"] == {"Ref": "AWS::AccountId"}
    arn = cond["ArnLike"]["aws:SourceArn"]["Fn::Join"]
    assert arn[0] == "" and arn[1] == ["arn:", {"Ref": "AWS::Partition"}, ":sagemaker:", {"Ref": "AWS::Region"}, ":",
                                        {"Ref": "AWS::AccountId"}, ":*"], arn


def test_default_synth_never_prompts_for_a_hub_token():
    """`cdk bootstrap`, `synth` and `diff` all run app.py; the token prompt must stay opt-in (-c hfTokenPrompt=true) or a
    fresh account is asked for a token before its bootstrap stack exists. Synth without HF_TOKEN_PROMPT, without CI and
    without usable credentials: a prompt that fires would surface as the NOTE app.py prints when it fails, or as a hang."""
    env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN_PROMPT", "CI") and not k.startswith("AWS_")}
    env.update(AWS_REGION="us-east-1", AWS_DEFAULT_REGION="us-east-1", AWS_EC2_METADATA_DISABLED="true",
               AWS_CONFIG_FILE="/dev/null", AWS_SHARED_CREDENTIALS_FILE="/dev/null")
    out = CDK / "cdk.out-test" / "default-prompt"
    r = subprocess.run(["cdk", "synth", "--quiet", "-o", str(out), "--app", f"{sys.executable} app.py"],
                       cwd=CDK, env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Hugging Face token" not in r.stderr and "hf token" not in r.stderr, r.stderr


def test_codebuild_uses_dockerfile_path_and_privileged(synth):
    template, _ = synth
    cb = next(r for r in template["Resources"].values() if r["Type"] == "AWS::CodeBuild::Project")
    spec = cb["Properties"]["Source"]["BuildSpec"]
    assert "-f sagemaker/Dockerfile" in spec and "BUILDKIT_INLINE_CACHE=1" in spec and "CODEBUILD_BUILD_SUCCEEDING" in spec
    assert cb["Properties"]["Environment"]["PrivilegedMode"] is True
    assert cb["Properties"]["TimeoutInMinutes"] == 120


def test_nag_report_has_no_unsuppressed_findings(synth):
    _, out = synth
    (report,) = glob.glob(str(out / "AwsSolutions-*NagReport.csv"))
    rows = list(csv.DictReader(open(report)))
    bad = [r for r in rows if r["Compliance"] == "Non-Compliant"]
    assert rows and not bad, bad


# --- committed plain-CloudFormation export (cdk/template/, for users without Node) ------------------------------------
# `cdk synth -c plainTemplate=1` is what cdk/export_template.py commits. The two file assets are keyed by content hash
# (trainer/ changes on every trainer edit; the kickoff handler on every aws-cdk-lib bump), so the comparison normalises
# hashes and CDK metadata away: only a real change to the stack fails it, and the failure names the regen command.
import difflib  # noqa: E402
import importlib.metadata  # noqa: E402
import importlib.util  # noqa: E402
import re  # noqa: E402

TEMPLATE_DIR = CDK / "template"
REGEN = "(cd cdk && . .venv/bin/activate && python export_template.py)"
HEX64 = re.compile(r"[0-9a-f]{64}")


def _export_module():
    """cdk/export_template.py, loaded by path (cdk/ is not a package). The synth flags live there and only there:
    a test-side copy could drift and make the comparison below fail with a metadata diff instead of a stale template."""
    spec = importlib.util.spec_from_file_location("export_template", CDK / "export_template.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PLAIN_FLAGS = _export_module().SYNTH_FLAGS


@pytest.fixture(scope="module")
def plain_synth():
    env = SYNTH_ENV
    out = CDK / "cdk.out-test" / "plain"  # under the git-ignored cdk.out-test/
    r = subprocess.run(["cdk", "synth", "--quiet", *PLAIN_FLAGS, "-o", str(out), "--app", f"{sys.executable} app.py"],
                       cwd=CDK, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    (template,) = glob.glob(str(out / "*infra.template.json"))
    (assets,) = glob.glob(str(out / "*infra.assets.json"))
    return json.load(open(template)), json.load(open(assets)), out


@pytest.fixture(scope="module")
def committed():
    # Read the template assets.json declares rather than globbing: a second <project>-infra.template.json (regenerated
    # under another project name, as the README suggests) must not turn every test here into a ValueError.
    assets = json.load(open(TEMPLATE_DIR / "assets.json"))
    template = REPO / assets["template"]
    assert template.is_file(), f"{assets['template']} missing; {REGEN}"
    return json.load(open(template)), assets


def normalise(template: dict) -> str:
    """Asset hashes -> placeholder; AWS::CDK::Metadata (and its Conditions.CDKMetadataAvailable companion, which a synth
    without --no-version-reporting emits) and aws:cdk:path / aws:asset:* metadata dropped."""
    t = json.loads(json.dumps(template))
    t["Resources"] = {k: r for k, r in t["Resources"].items() if r["Type"] != "AWS::CDK::Metadata"}
    t.get("Conditions", {}).pop("CDKMetadataAvailable", None)
    if not t.get("Conditions"):
        t.pop("Conditions", None)
    for r in t["Resources"].values():
        md = r.get("Metadata")
        if md:
            for key in [k for k in md if k == "aws:cdk:path" or k.startswith("aws:asset:")]:
                del md[key]
            if not md:
                del r["Metadata"]
    return HEX64.sub("<ASSET_HASH>", json.dumps(t, indent=1, sort_keys=True))


def test_plain_template_matches_committed_export(plain_synth, committed):
    fresh, _, _ = plain_synth
    stored, assets = committed
    a, b = normalise(stored), normalise(fresh)
    if a != b:
        diff = "\n".join(difflib.unified_diff(a.splitlines(), b.splitlines(), "committed", "fresh synth", lineterm="", n=2))
        have, want = importlib.metadata.version("aws-cdk-lib"), assets["generated_with"]["aws_cdk_lib"]
        why = (f"the stack changed since the last export (both on aws-cdk-lib {have})" if have == want
               else f"committed with aws-cdk-lib {want}, this synth uses {have}")
        pytest.fail(f"cdk/template/ is stale: {why}. Regenerate with {REGEN}\n{diff}")


def test_plain_template_has_no_bootstrap_rule(committed):
    stored, assets = committed
    text = json.dumps(stored)
    assert stored["Description"] == solution.stack_description()   # the export is the same install, deployed without Node
    assert "Rules" not in stored and "Conditions" not in stored and "Transform" not in stored
    assert set(stored["Parameters"]) == {assets["template_parameter"]} == {"AssetsBucketName"}
    for needle in ("BootstrapVersion", "cdk-bootstrap", "hnb659fds", "us-east-1"):
        assert needle not in text, needle
    counts = {}
    for res in stored["Resources"].values():
        counts[res["Type"]] = counts.get(res["Type"], 0) + 1
    assert counts == EXPECTED_TYPES and set(stored["Outputs"]) == EXPECTED_OUTPUTS
    # measure the file as committed (indent=2), i.e. what `aws cloudformation deploy --template-file` actually sends
    size = (REPO / assets["template"]).stat().st_size
    readme = (CDK / "template" / "README.md").read_text()
    if size >= 51_200:   # over the inline --template-file limit: the deploy command in the README must upload it via --s3-bucket
        assert re.search(r"^\s*--s3-bucket\s+\S", readme, re.M) and "`--s3-bucket` is required" in readme, \
            f"{size} bytes on disk: cdk/template/README.md must pass --s3-bucket in the deploy command and say it is required"
    else:
        assert "`--s3-bucket` is required" not in readme, "template fits inline again: drop the --s3-bucket requirement from the README"


def test_assets_json_lists_every_manifest_asset(plain_synth, committed):
    _, manifest, _ = plain_synth
    stored, assets = committed
    zips = {v["displayName"]: v for v in manifest["files"].values() if v["source"]["packaging"] == "zip"}
    listed = {a["display_name"]: a for a in assets["assets"]}
    assert set(listed) == set(zips), f"assets.json {sorted(listed)} vs manifest {sorted(zips)}; regenerate with {REGEN}"
    for name, a in listed.items():
        (dest,) = zips[name]["destinations"].values()
        assert a["bucket"] == dest["bucketName"] == "${AssetsBucketName}"
        assert a["s3_key"] == f"{a['id']}.zip" and a["packaging"] == "zip"
        assert (REPO / a["path"]).is_dir(), a["path"]
    # internally consistent with the committed template, hash-agnostic w.r.t. the fresh synth (trainer/ edits)
    assert {a["s3_key"] for a in assets["assets"]} == {h + ".zip" for h in set(HEX64.findall(json.dumps(stored)))}


def test_committed_handlers_match_synth(plain_synth, committed):
    """Both committed helper-Lambda code folders (build kickoff, S3 auto-delete) equal what aws-cdk-lib generates."""
    _, manifest, out = plain_synth
    _, assets = committed
    have, want = importlib.metadata.version("aws-cdk-lib"), assets["generated_with"]["aws_cdk_lib"]
    if have != want:
        pytest.skip(f"handler code tracks aws-cdk-lib: committed with {want}, running {have}; "
                    f"install cdk/requirements.txt or {REGEN}")
    handlers = [a for a in assets["assets"] if a["path"].startswith("cdk/template/assets/")]
    assert sorted(a["path"] for a in handlers) == ["cdk/template/assets/kickoff-handler/", "cdk/template/assets/s3-auto-delete-handler/"]
    for handler in handlers:
        assert handler["id"] in manifest["files"], f"handler asset {handler['id'][:12]} not in this synth; {REGEN}"
        fresh_dir = out / manifest["files"][handler["id"]]["source"]["path"]
        for f in handler["files"]:
            assert (REPO / handler["path"] / f).read_bytes() == (fresh_dir / f).read_bytes(), f


def test_requirements_pins_match_the_export(committed):
    """cdk/template/README.md promises the pins equal the export versions; this is what makes that true in CI."""
    _, assets = committed
    pins = dict(re.findall(r"^([A-Za-z0-9_.-]+)==(\S+)", (CDK / "requirements.txt").read_text(), re.M))
    want = assets["generated_with"]
    assert pins.get("aws-cdk-lib") == want["aws_cdk_lib"] and pins.get("cdk-nag") == want["cdk_nag"], \
        f"cdk/requirements.txt pins {pins} but cdk/template was exported with {want}; bump both together: {REGEN}"


@pytest.fixture(scope="module")
def retain_synth():
    """-c retainData=true: the bucket, the repository and the Hub token secret outlive the stack, and cdk-nag still passes."""
    env = SYNTH_ENV
    out = CDK / "cdk.out-test" / "retain"
    r = subprocess.run(["cdk", "synth", "--quiet", "-o", str(out), "-c", "retainData=true", "--app", f"{sys.executable} app.py"],
                       cwd=CDK, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    (template,) = glob.glob(str(out / "*infra.template.json"))
    return json.load(open(template)), out


def test_retain_data_true_keeps_bucket_repository_and_secret(retain_synth):
    t, out = retain_synth
    counts = {}
    for res in t["Resources"].values():
        counts[res["Type"]] = counts.get(res["Type"], 0) + 1
    counts.pop("AWS::CDK::Metadata", None)
    assert counts == EXPECTED_TYPES_RETAIN
    bucket = next(v for v in t["Resources"].values() if v["Type"] == "AWS::S3::Bucket")
    repo = next(v for v in t["Resources"].values() if v["Type"] == "AWS::ECR::Repository")
    assert bucket["DeletionPolicy"] == bucket["UpdateReplacePolicy"] == "Retain"
    assert repo["DeletionPolicy"] == repo["UpdateReplacePolicy"] == "Retain"
    assert repo["Properties"].get("EmptyOnDelete") is not True
    assert not any(tag["Key"] == "aws-cdk:auto-delete-objects" for tag in bucket["Properties"].get("Tags", []))
    policy = next(v for v in t["Resources"].values() if v["Type"] == "AWS::S3::BucketPolicy")
    assert policy["DeletionPolicy"] == policy["UpdateReplacePolicy"] == "Retain"    # the TLS-only policy stays with the bucket
    assert not any(v["Type"] == "Custom::S3AutoDeleteObjects" for v in t["Resources"].values())
    # the secret cleanup exists in both modes (so a mode flip on a deployed stack never sends it a Delete event) and is
    # RETAIN here: `cdk destroy` drops it without running the call
    secret_cleanup = next(v for v in t["Resources"].values() if v["Type"] == "Custom::HubTokenSecretCleanup")
    assert secret_cleanup["DeletionPolicy"] == "Retain"
    assert any(v["Type"] == "Custom::ImageParamsCleanup" for v in t["Resources"].values())   # stale image pointers go in both modes
    csvs = glob.glob(str(out / "AwsSolutions-*-NagReport.csv"))
    assert csvs, "cdk-nag report missing in retain-mode synth"
    rows = [r for r in csv.DictReader(open(csvs[0])) if r["Compliance"] == "Non-Compliant"]
    assert not rows, rows
