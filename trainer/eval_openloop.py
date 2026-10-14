#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Open-loop evaluation of a chunked policy on held-out LeRobot v3.0 episodes.

For every (strided) frame t of each held-out episode, the policy receives the
observation at t (every camera listed in meta/info.json plus observation.state)
and predicts a chunk of H absolute joint-position actions. Any camera count and
any action width are accepted. We compare against the ground-truth action chunk
a[t : t+H] and against a "hold" baseline that predicts the current state s[t]
for the whole horizon (zero motion). A model is only learning if it beats hold.

Reported metrics (JSON + stdout table):
  - per-dim MAE, model vs hold, aggregated over all joints and all grippers; for
    a 16-dim action vector additionally split into active-arm joints, idle-arm
    joints, active gripper and idle gripper, separately for left-arm and
    right-arm episodes.
  - gripper open/close timing (16-dim only): for each ground-truth gripper event
    (threshold crossing with hysteresis), does the chunk predicted 0.48 s before
    the event cross the threshold within +/-0.5 s of the true event time?

Dim layout: for a 16-dim action vector the 7+1+7+1 bimanual layout is ASSUMED
(0-6 left joints, 7 left gripper, 8-14 right joints, 15 right gripper) and the
arm-split / gripper-timing metrics are reported; any other width is scored per
dimension (all dims as joints, no arm split, no gripper timing).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

FPS = 25
LEFT_JOINTS = list(range(0, 7))
LEFT_GRIP = 7
RIGHT_JOINTS = list(range(8, 15))
RIGHT_GRIP = 15
TIMING_TOL_FRAMES = 12.5          # +/-0.5 s at 25 fps
TIMING_VANTAGE = 12               # predict from 0.48 s before the GT event


def detect_events(signal: np.ndarray, lo: float, hi: float):
    """Threshold-crossing events with hysteresis.

    Returns list of (frame_idx, kind) where kind is 'close' (falling below lo
    after having been above hi) or 'open' (rising above hi after below lo).
    """
    events = []
    state = "open" if signal[0] > (lo + hi) / 2 else "closed"
    for i in range(1, len(signal)):
        if state == "open" and signal[i] < lo:
            events.append((i, "close"))
            state = "closed"
        elif state == "closed" and signal[i] > hi:
            events.append((i, "open"))
            state = "open"
    return events


def crossing_time(pred: np.ndarray, thresh: float, kind: str):
    """First frame offset where pred crosses thresh in the event direction."""
    if kind == "close":
        hits = np.where(pred < thresh)[0]
    else:
        hits = np.where(pred > thresh)[0]
    return int(hits[0]) if len(hits) else None


def weights_file(ckpt_dir: str) -> str:
    """model.safetensors for a full checkpoint, adapter_model.safetensors for a PEFT (LoRA) checkpoint."""
    for name in ("model.safetensors", "adapter_model.safetensors"):
        if os.path.isfile(os.path.join(ckpt_dir, name)):
            return os.path.join(ckpt_dir, name)
    return os.path.join(ckpt_dir, "model.safetensors")


def load_policy(cls, ckpt_dir: str, device: str, peft_base: str = None):
    """from_pretrained for a full checkpoint; for a PEFT checkpoint (adapter_config.json, no model.safetensors) build the
    base policy from the staged Hub cache with the checkpoint's own config (obs_cam_keys, channel ids, dtype ...), attach
    the adapter, merge it, and rebuild a fresh policy from the merged weights (see _merged_state_dict)."""
    if os.path.isfile(os.path.join(ckpt_dir, "adapter_config.json")) and not os.path.isfile(os.path.join(ckpt_dir, "model.safetensors")):
        if not peft_base:
            raise SystemExit(f"{ckpt_dir} is a PEFT adapter checkpoint: pass --peft-base <base repo id> (registry base_checkpoint)")
        import gc

        from lerobot.configs.policies import PreTrainedConfig
        cfg = PreTrainedConfig.from_pretrained(ckpt_dir)
        cfg.device = device
        state = _merged_state_dict(cls, cfg, ckpt_dir, peft_base)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Built on `device` only now: the merge's objects are gone (a second 6B FastWAM in fp32 would not fit next to them on an
        # 80 GB card). Not built on the CPU and moved: FastWAM keeps its own `device` attribute for the prompt encoder, which
        # `.to()` does not update (a CPU build failed with a cuda/cpu mismatch in encode_prompt on the 2026-09-22 smoke).
        fresh = cls(cfg)
        missing, unexpected = fresh.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"WARNING: rebuilt merged policy: {len(missing)} missing, {len(unexpected)} unexpected keys: {missing[:3]} {unexpected[:3]}", file=sys.stderr)
        return fresh.to(device).eval()
    return cls.from_pretrained(ckpt_dir).to(device).eval()


def _merged_state_dict(cls, cfg, ckpt_dir: str, peft_base: str) -> dict:
    """Merge the adapter into its base in memory and return ONLY the merged weights (CPU tensors); the merged objects die here.

    PEFT replaces each wrapped module in the REGISTERED module tree with its trained copy. A policy that also keeps plain
    attribute aliases to those modules (FastWAM: `action_expert`, `video_expert`, MoT `_experts`, all set with
    object.__setattr__) keeps running the untouched originals through the aliases after merge_and_unload(). Measured on a
    fastwam_lora checkpoint 2026-09-22: the alias action expert differed from the registered one by 0.81, the reload of one
    checkpoint differed from its own eval by 176-251 degrees, and two reloads differed from each other. The caller rebuilds
    a fresh policy from these weights (what save_pretrained + from_pretrained do for the merged copy), which makes the
    in-memory merge identical to the merged copy."""
    from peft import PeftModel
    base = cls.from_pretrained(peft_base, config=cfg)
    merged = PeftModel.from_pretrained(base, ckpt_dir).merge_and_unload()
    return {k: v.detach().to("cpu").clone() for k, v in merged.state_dict().items()}


def predict_chunk(policy, batch, via_queues: bool = False, queue_image_key: str = None, chunk_first: bool = False, reset_per_call: bool = False):
    """One action chunk for a preprocessed batch. Most policies take the (B, [n_obs,] ...) batch directly. Queue policies
    (registry eval.history_via_queues: VQ-BeT, Multi-Task DiT, TD-MPC) read self._queues inside predict_action_chunk, so the
    history axis is fed frame by frame through populate_queues exactly as select_action would during a rollout:
      - cameras are stacked under observation.images (VQ-BeT, Multi-Task DiT) unless eval.queue_image_key names the single
        image key the policy queues instead (TD-MPC: observation.image);
      - every other preprocessed key (language tokens, attention masks) is passed through untouched;
      - eval.chunk_first transposes a (horizon, B, D) result to (B, horizon, D) (TD-MPC)."""
    if reset_per_call and hasattr(policy, "reset"):   # stateful autoregressive policies (LingBot-VA): every frame is a fresh episode start
        policy.reset()
    if not via_queues:
        out = policy.predict_action_chunk(batch)
        return out.transpose(0, 1) if chunk_first and isinstance(out, torch.Tensor) else out
    from lerobot.policies.utils import populate_queues
    from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
    policy.reset()
    state = batch[OBS_STATE]
    n_obs = state.shape[1] if state.ndim == 3 else 1
    image_keys = list(policy.config.image_features)
    for t in range(n_obs):
        frame = {OBS_STATE: state[:, t] if state.ndim == 3 else state}
        imgs = [batch[k][:, t] if batch[k].ndim == 5 else batch[k] for k in image_keys]
        if queue_image_key:
            frame[queue_image_key] = imgs[0]
        else:
            frame[OBS_IMAGES] = torch.stack(imgs, dim=-4)
        policy._queues = populate_queues(policy._queues, frame)
    extra = {k: v for k, v in batch.items() if k not in policy._queues and k not in image_keys and k != OBS_STATE}
    out = policy.predict_action_chunk({**extra, **frame})
    return out.transpose(0, 1) if chunk_first and isinstance(out, torch.Tensor) else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="pretrained policy dir")
    ap.add_argument("--policy", default="act",
                    help="lerobot policy type name (any type registered in lerobot.policies.factory)")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--repo-id", default="local/dataset",
                    help="repo id passed to LeRobotDataset for a local root; cosmetic")
    ap.add_argument("--episodes", type=int, nargs="+", required=True,
                    help="held-out episode ids (the trainer passes the ids from split.json)")
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--history-via-queues", action="store_true",
                    help="registry eval.history_via_queues: feed the observation history through the policy's queues (VQ-BeT)")
    ap.add_argument("--queue-image-key", default=None, help="registry eval.queue_image_key: single image key the policy queues (TD-MPC: observation.image)")
    ap.add_argument("--chunk-first", action="store_true", help="registry eval.chunk_first: predict_action_chunk returns (horizon, B, D)")
    ap.add_argument("--reset-per-call", action="store_true", help="registry eval.reset_per_call: policy.reset() before every prediction (stateful autoregressive policies)")
    ap.add_argument("--peft-base", default=None, help="base repo id for a PEFT adapter checkpoint (registry base_checkpoint)")
    ap.add_argument("--tolerance-s", type=float, default=1e-4, help="registry eval.tolerance_s: LeRobotDataset timestamp tolerance (15 fps datasets need ~2e-3)")
    ap.add_argument("--no-history", action="store_true", help="registry eval.no_history: observation_delta_indices are not an observation history (TD-MPC: future targets); feed one frame")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--save-preds", default=None, help="optional .npz of raw preds")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--video-backend", default=os.environ.get("VIDEO_BACKEND") or None,
                    help="lerobot video backend (torchcodec|pyav); default: $VIDEO_BACKEND or lerobot's choice")
    ap.add_argument("--n-samples", type=int, default=None,
                    help="rollouts to average per frame; default 1 for act/groot, "
                         "4 for xvla (flow matching draws fresh noise every call)")
    args = ap.parse_args()
    if args.n_samples is None:
        args.n_samples = 4 if args.policy == "xvla" else 1

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    PolicyCls = get_policy_class(args.policy)
    device = torch.device(args.device)
    policy = load_policy(PolicyCls, args.ckpt, args.device, args.peft_base)
    policy.to(device).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.ckpt,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    H = args.chunk
    # a checkpoint trained with a shorter chunk would silently shorten every record
    # while summary["chunk"] still claimed H, and would eat the timing vantage headroom
    ckpt_chunk = getattr(policy.config, "chunk_size", None)
    if ckpt_chunk is not None and ckpt_chunk < H:
        raise SystemExit(f"checkpoint chunk_size={ckpt_chunk} < --chunk={H}")

    # fps from the dataset itself; the module-level FPS is only the fallback
    try:
        fps = json.load(open(Path(args.dataset_root) / "meta" / "info.json"))["fps"]
    except Exception:
        fps = FPS
    globals()["FPS"] = fps
    globals()["TIMING_TOL_FRAMES"] = 0.5 * fps        # +/-0.5 s  (12.5 frames at 25 fps)
    globals()["TIMING_VANTAGE"] = int(round(0.48 * fps))  # 0.48 s before the event (12 at 25 fps)
    delta = {"action": [i / fps for i in range(H)]}
    # policies with an observation history (diffusion n_obs_steps=2, vqbet 5, ...) expect stacked past frames; lerobot
    # publishes the offsets as config.observation_delta_indices (None for single-frame policies such as ACT)
    obs_idx = None if args.no_history else getattr(policy.config, "observation_delta_indices", None)
    probe = LeRobotDataset(args.repo_id, root=args.dataset_root, episodes=args.episodes[:1], video_backend=args.video_backend, tolerance_s=args.tolerance_s)
    cam_keys = [k for k in probe.meta.camera_keys]
    if obs_idx:
        for k in cam_keys + ["observation.state"]:
            delta[k] = [i / fps for i in obs_idx]
        print(f"observation history: {len(obs_idx)} frames (offsets {list(obs_idx)})")
    ds = LeRobotDataset(args.repo_id, root=args.dataset_root,
                        episodes=args.episodes, delta_timestamps=delta,
                        video_backend=args.video_backend, tolerance_s=args.tolerance_s)

    print(f"dataset: {len(ds)} frames over episodes {args.episodes}; cams={cam_keys}")

    # dim layout: the 16-dim bimanual OpenArm layout gets the arm split + gripper timing;
    # anything else is scored generically (all dims as joints, no arm split, no gripper timing)
    D = int(ds.meta.features["action"]["shape"][0])
    if D == 16:
        LJ, LG, RJ, RG, bimanual = LEFT_JOINTS, LEFT_GRIP, RIGHT_JOINTS, RIGHT_GRIP, True
        dim_names = ([f"L_joint{i}" for i in range(7)] + ["L_gripper"]
                     + [f"R_joint{i}" for i in range(7)] + ["R_gripper"])
    else:
        print(f"WARNING: action dim {D} != 16 -- generic layout: all dims scored as joints, "
              "no left/right split, no gripper timing")
        LJ, LG, RJ, RG, bimanual = list(range(D)), None, [], None, False
        dim_names = [f"dim{i}" for i in range(D)]
    # a chunk predicted TIMING_VANTAGE frames before an event only covers H - vantage frames after it
    eff_tol = min(TIMING_TOL_FRAMES, H - TIMING_VANTAGE)

    # per-frame records: episode, t, |err| arrays for model and hold
    per_ep = {e: {"model_abs": [], "hold_abs": [], "t": [],
                  "pred_chunks": [], "gt_chunks": [], "state": [], "spread": []}
              for e in args.episodes}
    want_imgs = list(getattr(policy.config, "obs_cam_keys", None) or getattr(policy.config, "image_features", {}) or {})   # LingBot-VA reads obs_cam_keys
    checked_imgs = False

    with torch.no_grad():
        for i in range(0, len(ds), args.stride):
            item = ds[i]
            ep = int(item["episode_index"])
            gt = item["action"].numpy()                     # (H, 16)
            pad = item.get("action_is_pad")
            pad = pad.numpy() if pad is not None else np.zeros(H, bool)
            if pad.all():
                continue
            valid = ~pad

            obs = {"observation.state": item["observation.state"]}
            for k in cam_keys:
                obs[k] = item[k]
            if "task" in item:
                obs["task"] = item["task"]

            if obs_idx:
                # with a history axis the tensors are already 2-D/4-D and lerobot's add-batch-dimension step leaves them
                # alone, so (n_obs, D) would be read as (B, n_obs); add the batch axis ourselves
                obs = {k: (v.unsqueeze(0) if hasattr(v, "unsqueeze") else v) for k, v in obs.items()}
            batch = pre(obs)

            if not checked_imgs:
                # a mis-renamed camera key is NOT an error for xvla: the missing view is
                # replaced by an all-zero masked frame and training/eval proceed blind
                missing = [k for k in want_imgs if k not in obs]   # raw keys: GR00T's pack step replaces them in the batch
                if missing:
                    raise SystemExit(f"policy expects image keys absent from the observation: "
                                     f"{missing}; observation has {sorted(obs)}")
                checked_imgs = True

            frame_uid = int(item["index"]) if "index" in item else i
            samples = []
            for s in range(args.n_samples):
                # seed on the frame index, not the loop counter, so results do not
                # depend on --stride
                torch.manual_seed(args.seed + 1000 * s + frame_uid)
                chunk = predict_chunk(policy, batch, args.history_via_queues, args.queue_image_key, args.chunk_first, args.reset_per_call)
                if isinstance(chunk, torch.Tensor):
                    chunk = chunk.float()   # xvla runs in bf16; keeps the unnormalizer
                p = post(chunk)             # stats in fp32 and numpy() working
                if hasattr(p, "numpy"):
                    p = p.detach().cpu().float().numpy()
                samples.append(np.asarray(p))
            stack = np.stack(samples)
            pred = stack.mean(axis=0)[0][:H]
            spread = float(stack.std(axis=0).mean()) if args.n_samples > 1 else 0.0
            n = min(len(pred), H)
            state = item["observation.state"].numpy()
            if state.ndim == 2:          # (n_obs, D) history -> current frame is the last row
                state = state[-1]

            m_abs = np.abs(pred[:n][valid[:n]] - gt[:n][valid[:n]]).mean(axis=0)
            h_abs = np.abs(state[None, :] - gt[:n][valid[:n]]).mean(axis=0)

            rec = per_ep[ep]
            rec["model_abs"].append(m_abs)
            rec["hold_abs"].append(h_abs)
            rec["t"].append(int(item["frame_index"]))
            rec["pred_chunks"].append(pred[:n])
            rec["gt_chunks"].append(gt[:n])
            rec["state"].append(state)
            rec["spread"].append(spread)
            if i % 200 == 0:
                print(f"  {i}/{len(ds)}", flush=True)

    # ---- aggregate MAE ----
    def agg(eps, key, dims):
        if not dims:
            return None
        vals = [np.stack(per_ep[e][key])[:, dims].mean() for e in eps
                if per_ep[e][key]]
        return float(np.mean(vals)) if vals else None

    # classify episode arm by state variance of joint dims
    ep_arm = {}
    for e in args.episodes:
        if not bimanual or not per_ep[e]["state"]:
            ep_arm[e] = "L"          # generic layout: everything is scored under L_episodes / all
            continue
        states = np.stack(per_ep[e]["state"])
        ep_arm[e] = "L" if states[:, LJ].std(axis=0).mean() > \
                           states[:, RJ].std(axis=0).mean() else "R"
    left_eps = [e for e in args.episodes if ep_arm[e] == "L"]
    right_eps = [e for e in args.episodes if ep_arm[e] == "R"]
    print(f"episode arms: {ep_arm}")

    all_spread = [s for e in args.episodes for s in per_ep[e]["spread"]]
    summary = {"episodes": args.episodes, "ep_arm": ep_arm, "chunk": H,
               "stride": args.stride, "ckpt": str(args.ckpt),
               "policy": args.policy, "seed": args.seed,
               "n_samples": args.n_samples,
               "sample_spread_rad": float(np.mean(all_spread)) if all_spread else 0.0,
               "mae": {}}
    groups = {
        "L_episodes": (left_eps, LJ, LG, RJ, RG),
        "R_episodes": (right_eps, RJ, RG, LJ, LG),
    }
    for gname, (eps, aj, ag, ij, ig) in groups.items():
        agl = [ag] if ag is not None else []
        igl = [ig] if ig is not None else []
        summary["mae"][gname] = {
            "n_episodes": len(eps),
            "active_joints": {"model": agg(eps, "model_abs", aj),
                              "hold": agg(eps, "hold_abs", aj)},
            "active_gripper": {"model": agg(eps, "model_abs", agl),
                               "hold": agg(eps, "hold_abs", agl)},
            "idle_joints": {"model": agg(eps, "model_abs", ij),
                            "hold": agg(eps, "hold_abs", ij)},
            "idle_gripper": {"model": agg(eps, "model_abs", igl),
                             "hold": agg(eps, "hold_abs", igl)},
        }
    all_eps = args.episodes
    grips = [g for g in (LG, RG) if g is not None]
    summary["mae"]["all"] = {
        "all_joints": {"model": agg(all_eps, "model_abs", LJ + RJ),
                       "hold": agg(all_eps, "hold_abs", LJ + RJ)},
        "all_grippers": {"model": agg(all_eps, "model_abs", grips),
                         "hold": agg(all_eps, "hold_abs", grips)},
    }
    summary["layout"] = "openarm_bimanual_16" if bimanual else f"generic_{D}"

    # ---- true per-dim MAE, every dim ----
    # The grouped numbers above average 7 joints together, which hides a single
    # bad joint. This block is purely additive: same agg(), one dim at a time.
    summary["per_dim"] = {
        scope: {dim_names[d]: {"model": agg(eps, "model_abs", [d]),
                               "hold": agg(eps, "hold_abs", [d])}
                for d in range(D)}
        for scope, eps in (("all", all_eps), ("L_episodes", left_eps),
                           ("R_episodes", right_eps))
    }

    # ---- gripper timing ----
    timing = {"events": [], "hits": 0, "total": 0}
    for e in (args.episodes if bimanual else []):
        rec = per_ep[e]
        if not rec["t"]:
            continue
        grip_dim = LG if ep_arm[e] == "L" else RG
        # rebuild GT gripper signal from gt chunks at offset 0
        ts = np.array(rec["t"])
        order = np.argsort(ts)
        ts = ts[order]
        gt_sig = np.array([rec["gt_chunks"][j][0, grip_dim] for j in order])
        lo_v, hi_v = gt_sig.min(), gt_sig.max()
        rng = hi_v - lo_v
        if rng < 0.005:  # no meaningful gripper motion in this episode
            continue
        lo = lo_v + 0.35 * rng
        hi = lo_v + 0.65 * rng
        mid = lo_v + 0.5 * rng
        for idx, kind in detect_events(gt_sig, lo, hi):
            t_event = ts[idx]
            # prediction made TIMING_VANTAGE frames before the event
            want_t = t_event - TIMING_VANTAGE
            j_near = int(np.argmin(np.abs(ts - want_t)))
            if abs(ts[j_near] - want_t) > 2:
                continue
            pred_chunk = rec["pred_chunks"][order[j_near]][:, grip_dim]
            cross = crossing_time(pred_chunk, mid, kind)
            if cross is None:
                hit, err = False, None
            else:
                pred_event_t = ts[j_near] + cross
                err = float(pred_event_t - t_event)
                hit = abs(err) <= eff_tol
            timing["events"].append({"episode": e, "kind": kind,
                                     "t_gt": int(t_event),
                                     "err_frames": err, "hit": bool(hit)})
            timing["total"] += 1
            timing["hits"] += int(hit)
    summary["gripper_timing"] = {
        "hits": timing["hits"], "total": timing["total"],
        "accuracy": (timing["hits"] / timing["total"]) if timing["total"] else None,
        "tolerance_s": TIMING_TOL_FRAMES / FPS,
        "tolerance_frames_effective": float(eff_tol),
        "vantage_s": TIMING_VANTAGE / FPS,
        "events": timing["events"],
        "note": None if bimanual else f"skipped: action dim {D} is not the 16-dim bimanual layout",
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    if args.save_preds:
        np.savez_compressed(
            args.save_preds,
            **{f"ep{e}_pred": np.stack(per_ep[e]["pred_chunks"]) for e in all_eps if per_ep[e]["pred_chunks"]},
            **{f"ep{e}_gt": np.stack(per_ep[e]["gt_chunks"]) for e in all_eps if per_ep[e]["gt_chunks"]},
            **{f"ep{e}_t": np.array(per_ep[e]["t"]) for e in all_eps if per_ep[e]["t"]},
        )

    # ---- print table ----
    print("\n==== OPEN-LOOP EVAL ====")
    for gname in ["L_episodes", "R_episodes"]:
        g = summary["mae"][gname]
        print(f"\n{gname} (n={g['n_episodes']}):")
        for part in ["active_joints", "active_gripper", "idle_joints", "idle_gripper"]:
            m, h = g[part]["model"], g[part]["hold"]
            if m is None:
                continue
            beat = "BEATS hold" if m < h else "WORSE than hold"
            print(f"  {part:15s} model={m:.5f}  hold={h:.5f}  ({beat})")
    print("\nper-dim MAE over all held-out episodes (rad):")
    print(f"  {'dim':<10} {'model':>9} {'hold':>9} {'xhold':>6}")
    for d in range(D):
        pd_ = summary["per_dim"]["all"][dim_names[d]]
        m, h = pd_["model"], pd_["hold"]
        if m is None or h is None:
            continue
        ratio = f"{m/h:6.2f}" if h else "   n/a"
        print(f"  {dim_names[d]:<10} {m:9.5f} {h:9.5f} {ratio}"
              + ("   <-- worse than hold" if h and m >= h else ""))
    gt_ = summary["gripper_timing"]
    print(f"\ngripper timing: {gt_['hits']}/{gt_['total']} within +/-0.5 s"
          + (f" ({gt_['accuracy']*100:.0f}%)" if gt_["total"] else ""))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
