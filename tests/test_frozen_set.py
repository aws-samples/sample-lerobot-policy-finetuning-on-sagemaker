# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""The files that execute for POLICY=act inside the job are frozen once the image is validated.

Changing any of them re-opens validation: update FROZEN_SHA256 in the same change AND record the re-validation in
docs/RESULTS.md. Adding a policy YAML does not touch this set on purpose, with one exception: a non-stub entry whose
lerobot_type is new (or a stub promoted out of `status: stub`) must be added to POLICY_TYPES in
trainer/sagemaker/image_asserts.py, which is hashed here, so that change re-opens validation like any other
(test_image_asserts_cover_the_registry enforces the pairing).
"""
import ast
import hashlib
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
ACT_EXECUTION_SET = [
    "trainer/sagemaker/Dockerfile", "trainer/sagemaker/constraints.txt", "trainer/sagemaker/requirements.lock",
    "trainer/sagemaker/image_asserts.py", "trainer/sagemaker/train_entrypoint.sh",
    "trainer/train_policy.sh", "trainer/render_flags.py", "trainer/licence.py", "trainer/check_dataset.py",
    "trainer/downscale_plan.py", "trainer/prepare_dataset_360.sh", "trainer/eval_openloop.py", "trainer/summary.py",
    "trainer/publish.py", "trainer/smoke_checks.py", "trainer/reload_check.py", "trainer/policies/act.yaml",
]
# The hash of the validated tree; bump it in the same change as any edit to the files above, with a docs/RESULTS.md row.
FROZEN_SHA256 = "6a5dad9ceb20f6b0ccb55f434142ffc68653844d94d5a02bd0ee385c5778e573"  # 2026-09-23: licence headers on every source file (comment lines only); image_asserts.py imports every non-stub registry policy class (eo1, fastwam, lingbot_va, groot added; groot hard), the Dockerfile verifies the ResNet18 weights by full sha256 instead of file size, the eval.stochastic comment in train_entrypoint.sh rewritten; ACT path: the weight-file check only (same file, stricter test). Validated 2026-09-23 on sha256:d03dd0797805 (CPU ACT smoke with --expect-joints-mae, joints MAE 17.3817). Previous e6b0105c (2026-09-22, sha256:ba67992091ad, smoke-v1r, joints MAE 17.3817), 74c9199c (2026-09-21), 96994b9c (2026-09-18), 583b406a (2026-09-16, sha256:2cc7907c7dd2)  # pragma: allowlist secret


def frozen_set_hash() -> str:
    h = hashlib.sha256()
    for rel in ACT_EXECUTION_SET:
        h.update(rel.encode() + b"\0" + (REPO / rel).read_bytes() + b"\0")
    return h.hexdigest()


def test_every_frozen_file_exists():
    missing = [rel for rel in ACT_EXECUTION_SET if not (REPO / rel).is_file()]
    assert not missing, missing


def test_frozen_set_unchanged_since_validation():
    current = frozen_set_hash()
    assert current == FROZEN_SHA256, ("the POLICY=act execution set changed since the validated image; re-validate and update "
                                     f"FROZEN_SHA256 (now {current})")


def image_asserts_policy_types() -> set:
    """The POLICY_TYPES tuple of trainer/sagemaker/image_asserts.py, read without importing the module (it needs the image)."""
    tree = ast.parse((REPO / "trainer/sagemaker/image_asserts.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "POLICY_TYPES" for t in node.targets):
            return set(ast.literal_eval(node.value))
    raise AssertionError("POLICY_TYPES not found in image_asserts.py")


def test_image_asserts_cover_the_registry():
    """The build-time import check names every lerobot_type of a non-stub registry entry, and nothing else.

    The registry is COPYed into the image after the asserts run, so the tuple is kept by hand; this test keeps it in step.
    """
    registry = {}
    for f in sorted((REPO / "trainer/policies").glob("*.yaml")):
        d = yaml.safe_load(f.read_text())
        registry.setdefault(d["lerobot_type"], set()).add(d.get("status"))
    expected = {t for t, statuses in registry.items() if statuses != {"stub"}}
    assert image_asserts_policy_types() == expected, (
        f"missing from image_asserts.py: {sorted(expected - image_asserts_policy_types())}; "
        f"not in the registry or stub-only: {sorted(image_asserts_policy_types() - expected)}")
