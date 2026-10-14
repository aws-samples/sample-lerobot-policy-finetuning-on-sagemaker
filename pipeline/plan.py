# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""The plan finetune.sh shows before submitting anything: one row per selected policy with dataset, licence, instance,
steps, rate, estimated hours and dollars, and the uploads the stage step will do. Prices are approximate on-demand
On-demand list prices of one region (Asia Pacific, Tokyo; USD per hour, 2026-09) for planning only; the plan footer names the region; Spot is taken as half."""

PRICES_USD_H = {  # approximate, planning only
    "ml.g6e.xlarge": 2.5, "ml.g6e.2xlarge": 3.0, "ml.g6e.4xlarge": 5.3, "ml.g6e.8xlarge": 8.2, "ml.g6e.12xlarge": 14.0,
    "ml.g6e.16xlarge": 10.0, "ml.g6e.24xlarge": 20.0, "ml.g6e.48xlarge": 40.0,
    "ml.g6.xlarge": 1.2, "ml.g6.2xlarge": 1.4, "ml.g6.4xlarge": 1.9, "ml.g6.8xlarge": 2.9,
    "ml.g5.xlarge": 1.6, "ml.g5.2xlarge": 1.9, "ml.g5.4xlarge": 2.6, "ml.g4dn.2xlarge": 1.1,
    "ml.p5.4xlarge": 8.5, "ml.p5.48xlarge": 68.0, "ml.g7e.4xlarge": 9.0, "ml.c5.4xlarge": 0.9, "ml.m5.4xlarge": 1.0,
}


def price_per_hour(instance: str, spot: bool):
    p = PRICES_USD_H.get(instance)
    return None if p is None else (p * 0.5 if spot else p)


def policy_row(name: str, pol: dict, spec: dict, cfg: dict, dataset_name: str, instance: str, spot: bool, step_rate, instance_spec=None) -> dict:
    d = spec.get("defaults", {})
    steps = int(pol.get("steps", d.get("steps", 100000)))
    smoke = cfg.get("smoke", {}).get("enabled", True)
    lic = spec.get("licence") or {}
    cls = lic.get("class", "unknown")
    accepted = bool(pol.get("accept_licence")) if cls != "clean" else True
    approval = pol.get("approval_ref") if lic.get("requires_approval") or cls == "research_only" else "n/a"
    need = (spec.get("hardware") or {}).get("min_gpu_memory_gb")
    have = instance_spec.gpu_mem_gb if instance_spec else None
    est_h = (steps / step_rate) / 3600 + 0.35 if step_rate else None
    if est_h is not None and smoke:
        est_h += 0.25
    p = price_per_hour(instance, spot)
    return {"policy": name, "dataset": dataset_name, "licence": f"{cls}{'' if accepted else ' (accept_licence missing)'}" + ("" if approval == "n/a" or approval else " (approval_ref missing)"),
            "instance": instance + (" Spot" if spot else ""), "gpu": f"{have or '?'} GB / needs {need or '?'} GB" + (" TOO SMALL" if need and have and have < need else ""),
            "steps": steps, "steps_per_s": step_rate, "est_h": est_h, "est_usd": (est_h * p) if (est_h is not None and p) else None,
            "smoke": "yes" if smoke else "no", "blocked": (need and have and have < need) or not accepted or (approval not in ("n/a",) and not approval)}


def table(rows: list, uploads: list = ()) -> str:
    cols = [("policy", 15), ("dataset", 26), ("licence", 30), ("instance", 22), ("gpu", 24), ("steps", 7), ("steps_per_s", 8), ("est_h", 6), ("est_usd", 8), ("smoke", 5)]
    def fmt(r, k):
        v = r.get(k)
        if v is None: return "?"
        if k in ("steps_per_s",): return f"{v:.2f}"
        if k in ("est_h",): return f"{v:.1f}"
        if k in ("est_usd",): return f"~${v:.0f}"
        return str(v)
    head = " ".join(f"{k:<{w}}" for k, w in cols)
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(" ".join(f"{fmt(r, k)[:w]:<{w}}" for k, w in cols))
    total = sum(r["est_usd"] for r in rows if r.get("est_usd"))
    unknown = [r["policy"] for r in rows if r.get("est_usd") is None]
    lines.append(f"total estimate ~${total:.0f}" + (f" (+ {', '.join(unknown)}: no step rate yet, the smoke measures it)" if unknown else "") + "; prices approximate: ap-northeast-1 on-demand list prices, 2026-09, your region may differ; Spot counted at 50 %")
    for u in uploads:
        lines.append(f"upload: {u}")
    return "\n".join(lines)
