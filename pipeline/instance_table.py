# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Static facts about SageMaker training instance types (there is no API that returns GPU count/memory for ml.* types).

Used by preflight to refuse multi-GPU boxes (this trainer is one process on one GPU: a 4-GPU instance costs 2.5-4x for
the same speed), to warn about single-GPU boxes that are just bigger CPUs, and to pick the step-rate table row.
Values checked against the EC2 instance-type pages on 2026-09-10; `verified=False` rows are best effort.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Spec:
    gpu: Optional[str]      # GPU family name used in registry step_rate tables (L40S, A10G, T4, L4, H100, ...)
    gpus: int
    gpu_mem_gb: int
    vcpu: int
    verified: bool = True


def _family(prefix: str, gpu: str, mem: int, layout: dict, verified: bool = True) -> dict:
    return {f"ml.{prefix}.{size}": Spec(gpu, gpus, mem, vcpu, verified) for size, (gpus, vcpu) in layout.items()}


G_LAYOUT = {"xlarge": (1, 4), "2xlarge": (1, 8), "4xlarge": (1, 16), "8xlarge": (1, 32), "12xlarge": (4, 48),
            "16xlarge": (1, 64), "24xlarge": (4, 96), "48xlarge": (8, 192)}
TABLE: dict = {}
TABLE.update(_family("g4dn", "T4", 16, {"xlarge": (1, 4), "2xlarge": (1, 8), "4xlarge": (1, 16), "8xlarge": (1, 32), "12xlarge": (4, 48), "16xlarge": (1, 64)}))
TABLE.update(_family("g5", "A10G", 24, G_LAYOUT))
TABLE.update(_family("g6", "L4", 24, G_LAYOUT))
TABLE.update(_family("g6e", "L40S", 48, G_LAYOUT))
TABLE.update(_family("g7e", "RTX_PRO_6000", 96, {"2xlarge": (1, 8), "4xlarge": (1, 16), "8xlarge": (1, 32), "12xlarge": (2, 48), "24xlarge": (4, 96), "48xlarge": (8, 192)}, verified=False))
TABLE.update({
    "ml.p5.4xlarge": Spec("H100", 1, 80, 16, verified=False), "ml.p5.48xlarge": Spec("H100", 8, 80, 192),
    "ml.p5e.48xlarge": Spec("H200", 8, 141, 192), "ml.p5en.48xlarge": Spec("H200", 8, 141, 192),
    "ml.p4d.24xlarge": Spec("A100", 8, 40, 96), "ml.p4de.24xlarge": Spec("A100", 8, 80, 96),
    "ml.p3.2xlarge": Spec("V100", 1, 16, 8), "ml.p3.8xlarge": Spec("V100", 4, 16, 32), "ml.p3.16xlarge": Spec("V100", 8, 16, 64),
})
for _cpu, _vcpus in {"c5": {"xlarge": 4, "2xlarge": 8, "4xlarge": 16, "9xlarge": 36, "18xlarge": 72},
                     "m5": {"xlarge": 4, "2xlarge": 8, "4xlarge": 16, "12xlarge": 48, "24xlarge": 96}}.items():
    for _size, _n in _vcpus.items():
        TABLE[f"ml.{_cpu}.{_size}"] = Spec(None, 0, 0, _n)


def spec(instance_type: str) -> Optional[Spec]:
    return TABLE.get(instance_type)


def problems(instance_type: str, *, allow_multigpu: bool = False) -> tuple:
    """(fails, warns) for using this type with the single-process trainer."""
    s = spec(instance_type)
    if s is None:
        return [], [f"{instance_type}: not in the instance table; GPU count and memory unknown (add it to pipeline/instance_table.py)"]
    fails, warns = [], []
    if s.gpus > 1 and not allow_multigpu:
        fails.append(f"{instance_type} has {s.gpus} GPUs; this trainer uses one process on one GPU, so {s.gpus - 1} would idle "
                     f"(a multi-GPU box costs 2.5-4x for the same step rate). Pick a 1-GPU size or pass --allow-multigpu.")
    if s.gpus == 1 and s.vcpu >= 64:
        warns.append(f"{instance_type} is a single {s.gpu} with {s.vcpu} vCPUs: same step rate as the 4xlarge at a much higher price")
    if s.gpus >= 1 and s.vcpu <= 8:
        warns.append(f"{instance_type} has {s.vcpu} vCPUs: only {max(2, s.vcpu - 4)} dataloader workers for video decoding; expect a lower step rate")
    if not s.verified:
        warns.append(f"{instance_type}: GPU facts in the table are unverified; check the instance-type page before relying on them")
    return fails, warns


def quota_names(instance_type: str, spot: bool) -> list:
    return [f"{instance_type} for {'spot ' if spot else ''}training job usage"]


def step_rate(registry_spec: dict, instance_type: str) -> Optional[float]:
    """steps/s from the registry's hardware.step_rate table for this instance's GPU, else None."""
    s = spec(instance_type)
    table = (registry_spec.get("hardware") or {}).get("step_rate") or {}
    return table.get(s.gpu) if s and s.gpu else None
