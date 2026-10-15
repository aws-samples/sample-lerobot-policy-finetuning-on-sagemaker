# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Everything that can be checked before a dollar is spent, in the order the failures would otherwise surface.

    rows = run_preflight(cfg, stack, sess, policies, candidates=[...], allow_multigpu=False)

Rows are (level, check, message) with level ok | warn | fail; PreflightError is raised when any check fails.
Order matters: the licence gate and the instance table run before any AWS call, so a refused policy costs nothing.

    chosen = resolve_auto_instances({"act": False}, specs, client, region, allow_multigpu=False, log=print)

`--instance-type auto`: per policy, the first entry of the registry's hardware.instance_types that the instance table
accepts and whose quota (Spot or on-demand, as the policy runs) is above 0; PreflightError when no entry qualifies.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

from pipeline import instance_table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "trainer"))
import licence as licence_mod  # noqa: E402  (trainer/licence.py: same gate as inside the job)

class PreflightError(RuntimeError):
    pass


def _mask(arn: str) -> str:
    return re.sub(r"\d{12}", "<acct>", arn or "")


def local_checks(cfg: dict, policies: dict, candidates: list, *, allow_multigpu: bool = False, allow_stubs: bool = False) -> list:
    """Offline checks: licence gate per enabled policy, instance-table sanity for every candidate type."""
    rows = []
    for name, pol in cfg["policies"].items():
        if not pol.get("enabled"):
            continue
        spec = policies.get(name)
        if spec is None:
            rows.append(("fail", "registry", f"policies.{name}: no trainer/policies/{name}.yaml"))
            continue
        if spec.get("status") == "stub":
            rows.append(("warn" if allow_stubs else "fail", "registry",
                         f"{name} is a stub in this release{' (running anyway: --allow-stubs)' if allow_stubs else ''}: {spec.get('stub_reason', 'not runnable yet')}"))
        ok, msg = licence_mod.gate(spec, accept=bool(pol.get("accept_licence")), approval_ref=pol.get("approval_ref"))
        rows.append(("ok" if ok else "fail", "licence", msg))
    for it in candidates:
        fails, warns = instance_table.problems(it, allow_multigpu=allow_multigpu)
        rows += [("fail", "instance", f) for f in fails] + [("warn", "instance", w) for w in warns]
    return rows


def _quota_flags(instance_type: str, spot: bool) -> set:
    """Quota names to check: GPU trainers need the quota of the capacity they use (smoke and training share it: Spot or
    on-demand, e.g. a Spot-only p5 quota is enough for a Spot run); CPU staging / smoke jobs always run on-demand."""
    s = instance_table.spec(instance_type)
    return {spot} if (s and s.gpu) else {False}


def aws_checks(sess, stack: dict, cfg: dict, candidates: list, *, spot: bool, image_uri: Optional[str], client=None,
               training_plan_arn: Optional[str] = None) -> list:
    from pipeline.aws import client as mk
    client = client or (lambda name: mk(sess, name))
    rows = []
    region = sess.region_name
    ident = client("sts").get_caller_identity()
    rows.append(("ok", "credentials", f"{_mask(ident['Arn'])} in {region}"))
    s3 = client("s3")
    loc = s3.get_bucket_location(Bucket=stack["bucket"]).get("LocationConstraint") or "us-east-1"
    if loc != region:
        rows.append(("fail", "region", f"bucket is in {loc} but the session/stack region is {region}: SageMaker requires the dataset bucket in the job region"))
    else:
        rows.append(("ok", "region", f"bucket, stack and session all in {region}"))
    if not image_uri:
        rows.append(("fail", "image", "no image: the stack has not published /<project>/image_uri (is the deploy-time build finished?)"))
    elif "@sha256:" not in image_uri:
        rows.append(("fail", "image", f"image must be digest-pinned: {image_uri}"))
    else:
        repo_name = image_uri.split("/", 1)[1].split("@")[0]
        if not image_uri.startswith(stack["ecr_repo_uri"]):
            rows.append(("warn", "image", "image is not in this stack's ECR repo"))
        img_region = image_uri.split(".dkr.ecr.")[1].split(".")[0]
        if img_region != region:
            rows.append(("fail", "region", f"image lives in {img_region}, jobs run in {region}"))
        try:
            client("ecr").describe_images(repositoryName=repo_name, imageIds=[{"imageDigest": image_uri.split("@")[1]}])
            rows.append(("ok", "image", f"{repo_name}@{image_uri.split('@sha256:')[1][:12]}... present"))
        except Exception as e:
            rows.append(("fail", "image", f"image digest not found in ECR: {type(e).__name__}"))
    # training plan (reserved capacity): the GPU jobs run on it, so its status and instance type replace their quota check
    on_plan = set()
    if training_plan_arn:
        plan_name = training_plan_arn.rsplit("/", 1)[-1]
        plan_region = training_plan_arn.split(":")[3]
        if plan_region != region:
            rows.append(("fail", "plan", f"training plan {plan_name} is in {plan_region}, jobs run in {region}"))
        else:
            try:
                tp = client("sagemaker").describe_training_plan(TrainingPlanName=plan_name)
                types = {rc.get("InstanceType") for rc in tp.get("ReservedCapacitySummaries") or []}
                status = tp.get("Status")
                if status not in ("Scheduled", "Active"):
                    rows.append(("fail", "plan", f"training plan {plan_name} is {status}; a job can use it only while Scheduled or Active"))
                elif "training-job" not in (tp.get("TargetResources") or ["training-job"]):
                    rows.append(("fail", "plan", f"training plan {plan_name} targets {tp.get('TargetResources')}, not training jobs"))
                else:
                    gpu = [it for it in candidates if (instance_table.spec(it) or type("S", (), {"gpu": None})).gpu]
                    wrong = [it for it in gpu if it not in types]
                    if wrong:
                        rows.append(("fail", "plan", f"training plan {plan_name} reserves {sorted(types)}, the run needs {wrong}"))
                    else:
                        on_plan = set(gpu)
                        rows.append(("ok", "plan", f"{plan_name}: {status}, {sorted(types)}, {tp.get('AvailableInstanceCount', '?')} instance(s) available"
                                     + (f", ends {tp['EndTime']:%Y-%m-%d %H:%M} UTC" if hasattr(tp.get("EndTime"), "strftime") else "")))
            except Exception as e:
                rows.append(("fail", "plan", f"cannot describe training plan {plan_name}: {type(e).__name__}"))
    # quotas: both names for every candidate, one paginated pass
    wanted = {}
    for it in candidates:
        if it in on_plan:
            rows.append(("ok", "quota", f"{it}: reserved capacity via the training plan (no on-demand quota needed)"))
            continue
        for spot_flag in _quota_flags(it, spot):   # CPU jobs (smoke, hub staging) always run on-demand
            for qn in instance_table.quota_names(it, spot_flag):
                wanted[qn] = it
    try:
        found = quota_values(client, set(wanted))
        for qn, it in wanted.items():
            v = found.get(qn)
            if v is None:
                rows.append(("warn", "quota", f"{qn}: quota not listed in {region}"))
            elif v < 1:
                rows.append(("fail", "quota", f"{qn} = 0 in {region}: request an increase (Service Quotas > Amazon SageMaker) before submitting"))
            else:
                rows.append(("ok", "quota", f"{qn} = {v:g}"))
    except Exception as e:
        rows.append(("warn", "quota", f"could not read service quotas ({type(e).__name__}); a quota of 0 fails at submit time"))
    return rows


def quota_values(client, names: set) -> dict:
    """QuotaName -> Value for the SageMaker quotas in `names`, one paginated pass; a name the region does not list is absent."""
    found = {}
    names = set(names)
    if not names:
        return found
    for page in client("service-quotas").get_paginator("list_service_quotas").paginate(ServiceCode="sagemaker"):
        for q in page["Quotas"]:
            if q["QuotaName"] in names:
                found[q["QuotaName"]] = q["Value"]
        if len(found) == len(names):
            break
    return found


def _quota_name(instance_type: str, spot: bool) -> str:
    return instance_table.quota_names(instance_type, next(iter(_quota_flags(instance_type, spot))))[0]


def auto_instance(spec: dict, spot: bool, quotas: Optional[dict], *, allow_multigpu: bool = False) -> tuple:
    """Walk the registry entry's hardware.instance_types in order and pick the first usable one.

    Skipped: a type the instance table refuses (multi-GPU without allow_multigpu), a type not in the table (GPU facts
    unknown), and a type whose quota is 0 or not listed. `quotas` is QuotaName -> Value from quota_values(), or None when
    the quotas could not be read: then the first type the table accepts is taken. Returns (chosen, quota_value, skipped)
    with skipped = ["<type>: <why>", ...]; chosen is None when every type was skipped.
    """
    skipped = []
    for it in (spec.get("hardware") or {}).get("instance_types") or []:
        fails, _ = instance_table.problems(it, allow_multigpu=allow_multigpu)
        if instance_table.spec(it) is None:
            skipped.append(f"{it}: not in the instance table (GPU count and memory unknown; add it to pipeline/instance_table.py)"); continue
        if fails:
            skipped.append(f"{it}: {instance_table.spec(it).gpus} GPUs, refused without --allow-multigpu"); continue
        if quotas is None:
            return it, None, skipped
        qn = _quota_name(it, spot)
        v = quotas.get(qn)
        if v is None:
            skipped.append(f"{it}: quota not listed in this region ({qn})"); continue
        if v < 1:
            skipped.append(f"{it}: quota 0 ({qn})"); continue
        return it, v, skipped
    return None, None, skipped


def resolve_auto_instances(requests: dict, specs: dict, client, region: str, *, allow_multigpu: bool = False, log=print) -> dict:
    """policy -> instance type for the policies whose instance type is 'auto'; `requests` is policy -> Spot flag.

    One quota read covers every candidate of every policy. Raises PreflightError (exit 4 in run.py) when a policy has no
    type with quota, naming each type, its quota value and the quota to request.
    """
    names = set()
    for name, spot in requests.items():
        for it in (specs[name].get("hardware") or {}).get("instance_types") or []:
            if instance_table.spec(it) is not None:
                names.add(_quota_name(it, spot))
    try:
        quotas = quota_values(client, names)
    except Exception as e:   # AccessDenied, no endpoint: the submit still fails on a quota of 0, preflight repeats the warning
        quotas = None
        log(f"  instance auto: could not read service quotas ({type(e).__name__}); taking the first type the instance rules accept for each policy")
    chosen, failed = {}, []
    for name, spot in requests.items():
        if not (specs[name].get("hardware") or {}).get("instance_types"):
            failed.append(f"{name}: trainer/policies/{name}.yaml lists no hardware.instance_types; name a type in project_config.json"); continue
        it, value, skipped = auto_instance(specs[name], spot, quotas, allow_multigpu=allow_multigpu)
        if it is None:
            failed.append(f"{name}: every registry instance type was skipped in {region}: " + "; ".join(skipped)); continue
        chosen[name] = it
        q = "quota unread" if value is None else f"quota {value:g}"
        log(f"  instance auto: {name}: {it} ({q})" + (f" after skipping {', '.join(skipped)}" if skipped else ""))
    if failed:
        raise PreflightError("no instance type with quota:\n  " + "\n  ".join(failed)
                             + "\n  request an increase for one of the quotas named above (Service Quotas > Amazon SageMaker) or set the type in project_config.json")
    return chosen


def run_preflight(cfg: dict, stack: dict, sess, policies: dict, *, candidates: list, spot: bool = False,
                  image_uri: Optional[str] = None, allow_multigpu: bool = False, allow_stubs: bool = False, log=print, client=None,
                  training_plan_arn: Optional[str] = None) -> list:
    rows = local_checks(cfg, policies, candidates, allow_multigpu=allow_multigpu, allow_stubs=allow_stubs)
    if not any(lvl == "fail" for lvl, _, _ in rows):
        rows += aws_checks(sess, stack, cfg, candidates, spot=spot, image_uri=image_uri, client=client, training_plan_arn=training_plan_arn)
        if any(a.get("gated") for sp in policies.values() for a in (sp.get("hub_artifacts") or [])):
            # Hugging Face token: reported here, prompted for by run.py right before the staging job that needs it
            from pipeline import hf_token as _hf
            from pipeline.aws import client as _mk
            st = _hf.status((client or (lambda n: _mk(sess, n)))("secretsmanager"), _hf.secret_name(cfg))
            rows.append(("ok" if st["state"] == "ok" else "warn", "hf token",
                         f"{_hf.secret_name(cfg)}: {st['state']}" + (f" (user {st['user']})" if st.get("user") else "; asked for before the staging job (hidden input; a dry run only reports it)")))
    for lvl, check, msg in rows:
        log(f"  {lvl.upper():4s} {check:12s} {msg}")
    fails = [m for lvl, _, m in rows if lvl == "fail"]
    if fails:
        raise PreflightError(f"preflight found {len(fails)} problem(s): fix them before spending money")
    return rows
