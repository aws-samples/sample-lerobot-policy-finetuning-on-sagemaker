# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""step_rate_from_log against lerobot's real record shapes.

The four excerpts are consecutive mid-run lines from the 2026-09-11 VLA fine-tunes on ml.g6e.4xlarge (trailing per-policy
metrics trimmed). Every label past step 1000 is rounded to whole thousands (step:10K for steps 9600..10400), which made
the previous parser return 0.0 for every run longer than 1000 steps; the tqdm counter that shares the line carries the
exact step. Expected rates are the review's steady-state steps/s from the full logs (first-to-last INFO timestamp span):
4.58, 2.25, 1.23, 0.93 (5 % tolerance; the excerpts are 700-2200-step slices, and the median that step_rate_from_log
returns snaps to the modal 1 s-resolution window, so it sits within about 2 % of the span rate).
"""
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "trainer"))
import summary  # noqa: E402

SMOLVLA = """Training:  47%|████▋     | 9400/20000 [36:34<37:16,  4.74step/s]INFO 2026-09-11 15:49:38 ot_train.py:641 step:9K smpl:150K ep:639 epch:14.19 loss:0.082 grdn:1.240 lr:5.7e-05 updt_s:0.210 data_s:0.008 smp/s:73
Training:  48%|████▊     | 9600/20000 [37:18<38:19,  4.52step/s]INFO 2026-09-11 15:50:21 ot_train.py:641 step:10K smpl:154K ep:652 epch:14.50 loss:0.082 grdn:1.257 lr:5.5e-05 updt_s:0.212 data_s:0.003 smp/s:74
Training:  49%|████▉     | 9800/20000 [38:01<36:58,  4.60step/s]INFO 2026-09-11 15:51:05 ot_train.py:641 step:10K smpl:157K ep:666 epch:14.80 loss:0.081 grdn:1.191 lr:5.4e-05 updt_s:0.213 data_s:0.003 smp/s:74
Training:  50%|█████     | 10000/20000 [38:46<35:56,  4.64step/s]INFO 2026-09-11 15:51:49 ot_train.py:641 step:10K smpl:160K ep:680 epch:15.10 loss:0.080 grdn:1.182 lr:5.2e-05 updt_s:0.210 data_s:0.009 smp/s:73
Training:  51%|█████     | 10200/20000 [39:31<35:35,  4.59step/s]INFO 2026-09-11 15:52:34 ot_train.py:641 step:10K smpl:163K ep:693 epch:15.40 loss:0.076 grdn:1.111 lr:5.0e-05 updt_s:0.212 data_s:0.003 smp/s:74
Training:  52%|█████▏    | 10400/20000 [40:14<34:16,  4.67step/s]INFO 2026-09-11 15:53:17 ot_train.py:641 step:10K smpl:166K ep:707 epch:15.70 loss:0.076 grdn:1.139 lr:4.9e-05 updt_s:0.211 data_s:0.003 smp/s:75
Training:  53%|█████▎    | 10600/20000 [40:57<31:02,  5.05step/s]INFO 2026-09-11 15:54:00 ot_train.py:641 step:11K smpl:170K ep:720 epch:16.01 loss:0.075 grdn:1.136 lr:4.7e-05 updt_s:0.208 data_s:0.003 smp/s:76
Training:  54%|█████▍    | 10800/20000 [41:42<33:31,  4.57step/s]INFO 2026-09-11 15:54:45 ot_train.py:641 step:11K smpl:173K ep:734 epch:16.31 loss:0.071 grdn:1.101 lr:4.6e-05 updt_s:0.215 data_s:0.009 smp/s:72
Training:  55%|█████▌    | 11000/20000 [42:25<32:33,  4.61step/s]INFO 2026-09-11 15:55:28 ot_train.py:641 step:11K smpl:176K ep:747 epch:16.61 loss:0.070 grdn:1.145 lr:4.4e-05 updt_s:0.211 data_s:0.003 smp/s:75
Training:  56%|█████▌    | 11200/20000 [43:09<31:33,  4.65step/s]INFO 2026-09-11 15:56:12 ot_train.py:641 step:11K smpl:179K ep:761 epch:16.91 loss:0.069 grdn:1.092 lr:4.3e-05 updt_s:0.213 data_s:0.003 smp/s:74
Training:  57%|█████▋    | 11400/20000 [43:53<30:40,  4.67step/s]INFO 2026-09-11 15:56:56 ot_train.py:641 step:11K smpl:182K ep:775 epch:17.21 loss:0.070 grdn:1.110 lr:4.1e-05 updt_s:0.210 data_s:0.008 smp/s:73
Training:  58%|█████▊    | 11600/20000 [44:36<29:52,  4.69step/s]INFO 2026-09-11 15:57:39 ot_train.py:641 step:12K smpl:186K ep:788 epch:17.52 loss:0.066 grdn:1.125 lr:4.0e-05 updt_s:0.211 data_s:0.003 smp/s:75
"""
XVLA = """Training:  49%|████▉     | 14700/30000 [1:51:09<1:52:11,  2.27step/s]INFO 2026-09-11 17:04:20 ot_train.py:641 step:15K smpl:235K ep:999 epch:22.20 loss:0.007 grdn:0.375 lr:5.7e-06 updt_s:0.434 data_s:0.004 smp/s:37
Training:  49%|████▉     | 14800/30000 [1:51:53<1:51:20,  2.28step/s]INFO 2026-09-11 17:05:04 ot_train.py:641 step:15K smpl:237K ep:1K epch:22.35 loss:0.006 grdn:0.330 lr:5.6e-06 updt_s:0.433 data_s:0.004 smp/s:37
Training:  50%|████▉     | 14900/30000 [1:52:37<1:50:33,  2.28step/s]INFO 2026-09-11 17:05:47 ot_train.py:641 step:15K smpl:238K ep:1K epch:22.50 loss:0.006 grdn:0.364 lr:5.6e-06 updt_s:0.433 data_s:0.004 smp/s:37
Training:  50%|█████     | 15000/30000 [1:53:21<1:49:53,  2.28step/s]INFO 2026-09-11 17:06:32 ot_train.py:641 step:15K smpl:240K ep:1K epch:22.65 loss:0.005 grdn:0.285 lr:5.5e-06 updt_s:0.434 data_s:0.004 smp/s:36
Training:  50%|█████     | 15100/30000 [1:54:17<1:49:46,  2.26step/s]INFO 2026-09-11 17:07:28 ot_train.py:641 step:15K smpl:242K ep:1K epch:22.80 loss:0.005 grdn:0.342 lr:5.5e-06 updt_s:0.437 data_s:0.005 smp/s:36
Training:  51%|█████     | 15200/30000 [1:55:02<1:48:29,  2.27step/s]INFO 2026-09-11 17:08:13 ot_train.py:641 step:15K smpl:243K ep:1K epch:22.95 loss:0.006 grdn:0.328 lr:5.4e-06 updt_s:0.434 data_s:0.004 smp/s:37
Training:  51%|█████     | 15300/30000 [1:55:46<1:48:20,  2.26step/s]INFO 2026-09-11 17:08:57 ot_train.py:641 step:15K smpl:245K ep:1K epch:23.10 loss:0.006 grdn:0.405 lr:5.4e-06 updt_s:0.431 data_s:0.012 smp/s:36
"""
PI0 = """Training:  48%|████▊     | 9600/20000 [2:12:48<2:16:43,  1.27step/s]INFO 2026-09-11 17:42:46 ot_train.py:641 step:10K smpl:77K ep:326 epch:7.25 loss:0.082 grdn:1.637 lr:1.5e-05 updt_s:0.792 data_s:0.002 smp/s:10
Training:  48%|████▊     | 9700/20000 [2:14:08<2:15:34,  1.27step/s]INFO 2026-09-11 17:44:06 ot_train.py:641 step:10K smpl:78K ep:330 epch:7.32 loss:0.070 grdn:1.541 lr:1.4e-05 updt_s:0.794 data_s:0.002 smp/s:10
Training:  49%|████▉     | 9800/20000 [2:15:27<2:14:06,  1.27step/s]INFO 2026-09-11 17:45:25 ot_train.py:641 step:10K smpl:78K ep:333 epch:7.40 loss:0.071 grdn:1.546 lr:1.4e-05 updt_s:0.794 data_s:0.002 smp/s:10
Training:  50%|████▉     | 9900/20000 [2:16:47<2:13:38,  1.26step/s]INFO 2026-09-11 17:46:45 ot_train.py:641 step:10K smpl:79K ep:336 epch:7.47 loss:0.078 grdn:1.703 lr:1.4e-05 updt_s:0.795 data_s:0.002 smp/s:10
Training:  50%|█████     | 10000/20000 [2:18:07<2:14:02,  1.24step/s]INFO 2026-09-11 17:48:05 ot_train.py:641 step:10K smpl:80K ep:340 epch:7.55 loss:0.076 grdn:1.607 lr:1.4e-05 updt_s:0.792 data_s:0.002 smp/s:10
Training:  50%|█████     | 10100/20000 [2:19:47<2:12:24,  1.25step/s]INFO 2026-09-11 17:49:45 ot_train.py:641 step:10K smpl:81K ep:343 epch:7.63 loss:0.075 grdn:1.516 lr:1.4e-05 updt_s:0.791 data_s:0.003 smp/s:10
Training:  51%|█████     | 10200/20000 [2:21:06<2:09:05,  1.27step/s]INFO 2026-09-11 17:51:04 ot_train.py:641 step:10K smpl:82K ep:347 epch:7.70 loss:0.074 grdn:1.662 lr:1.3e-05 updt_s:0.790 data_s:0.002 smp/s:10
Training:  52%|█████▏    | 10300/20000 [2:22:26<2:07:40,  1.27step/s]INFO 2026-09-11 17:52:24 ot_train.py:641 step:10K smpl:82K ep:350 epch:7.78 loss:0.074 grdn:1.535 lr:1.3e-05 updt_s:0.795 data_s:0.002 smp/s:10
Training:  52%|█████▏    | 10400/20000 [2:23:45<2:05:35,  1.27step/s]INFO 2026-09-11 17:53:43 ot_train.py:641 step:10K smpl:83K ep:353 epch:7.85 loss:0.073 grdn:1.575 lr:1.3e-05 updt_s:0.794 data_s:0.002 smp/s:10
"""
PI05 = """Training:  48%|████▊     | 9600/20000 [2:54:46<3:03:24,  1.06s/step]INFO 2026-09-11 18:25:24 ot_train.py:641 step:10K smpl:77K ep:326 epch:7.25 loss:0.030 grdn:0.304 lr:1.5e-05 updt_s:1.050 data_s:0.004 smp/s:8
Training:  48%|████▊     | 9700/20000 [2:56:32<3:00:44,  1.05s/step]INFO 2026-09-11 18:27:10 ot_train.py:641 step:10K smpl:78K ep:330 epch:7.32 loss:0.029 grdn:0.291 lr:1.4e-05 updt_s:1.054 data_s:0.004 smp/s:8
Training:  49%|████▉     | 9800/20000 [2:58:17<3:01:15,  1.07s/step]INFO 2026-09-11 18:28:55 ot_train.py:641 step:10K smpl:78K ep:333 epch:7.40 loss:0.028 grdn:0.292 lr:1.4e-05 updt_s:1.047 data_s:0.004 smp/s:8
Training:  50%|████▉     | 9900/20000 [3:00:03<2:55:22,  1.04s/step]INFO 2026-09-11 18:30:41 ot_train.py:641 step:10K smpl:79K ep:336 epch:7.47 loss:0.028 grdn:0.294 lr:1.4e-05 updt_s:1.050 data_s:0.004 smp/s:8
Training:  50%|█████     | 10000/20000 [3:01:48<2:54:13,  1.05s/step]INFO 2026-09-11 18:32:26 ot_train.py:641 step:10K smpl:80K ep:340 epch:7.55 loss:0.028 grdn:0.291 lr:1.4e-05 updt_s:1.046 data_s:0.004 smp/s:8
Training:  50%|█████     | 10100/20000 [3:03:55<2:57:46,  1.08s/step]INFO 2026-09-11 18:34:34 ot_train.py:641 step:10K smpl:81K ep:343 epch:7.63 loss:0.028 grdn:0.277 lr:1.4e-05 updt_s:1.053 data_s:0.004 smp/s:8
Training:  51%|█████     | 10200/20000 [3:05:41<2:50:55,  1.05s/step]INFO 2026-09-11 18:36:19 ot_train.py:641 step:10K smpl:82K ep:347 epch:7.70 loss:0.027 grdn:0.291 lr:1.3e-05 updt_s:1.048 data_s:0.004 smp/s:8
Training:  52%|█████▏    | 10300/20000 [3:07:26<2:51:30,  1.06s/step]INFO 2026-09-11 18:38:05 ot_train.py:641 step:10K smpl:82K ep:350 epch:7.78 loss:0.029 grdn:0.292 lr:1.3e-05 updt_s:1.053 data_s:0.004 smp/s:8
Training:  52%|█████▏    | 10400/20000 [3:09:12<2:48:01,  1.05s/step]INFO 2026-09-11 18:39:50 ot_train.py:641 step:10K smpl:83K ep:353 epch:7.85 loss:0.026 grdn:0.288 lr:1.3e-05 updt_s:1.052 data_s:0.004 smp/s:8
"""
EXPECTED = {"smolvla": (SMOLVLA, 4.58), "xvla": (XVLA, 2.25), "pi0": (PI0, 1.23), "pi05": (PI05, 0.93)}
TQDM_PREFIX = re.compile(r"^Training:.*?\]\s*(?=INFO )", re.M)


def _rate(tmp_path, text):
    p = tmp_path / "train.log"
    p.write_text(text)
    return summary.step_rate_from_log(str(p))


@pytest.mark.parametrize("policy", sorted(EXPECTED))
def test_real_long_run_excerpts_with_the_tqdm_counter(tmp_path, policy):
    text, expected = EXPECTED[policy]
    rate, last = _rate(tmp_path, text)
    assert rate is not None and rate > 0, "the rounded step:NK labels must not yield 0.0"
    assert abs(rate - expected) / expected < 0.05, (policy, rate, expected)
    assert abs(last["timer_steps_per_s"] - expected) / expected < 0.05           # 1 / (updt_s + data_s) cross-check


@pytest.mark.parametrize("policy", sorted(EXPECTED))
def test_real_long_run_excerpts_without_the_tqdm_counter(tmp_path, policy):
    """Labels only (stdout and stderr not interleaved): distinct-label windows, or the per-step timers when the excerpt
    never changes label (xvla, pi0, pi0.5 stay at one label for these 700-800 steps)."""
    text, expected = EXPECTED[policy]
    stripped = TQDM_PREFIX.sub("", text)
    assert "Training:" not in stripped and "INFO 2026" in stripped
    rate, _ = _rate(tmp_path, stripped)
    assert rate is not None and rate > 0
    assert abs(rate - expected) / expected < 0.05, (policy, rate, expected)


def _synth(steps, log_freq, seconds_per_window, start="2026-09-11 15:00:00", stamp=True, tqdm=False, lag=0, updt_s=0.090):
    import datetime as dt
    t0 = dt.datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    lines = []
    for i, s in enumerate(range(log_freq, steps + 1, log_freq)):
        label = f"{s / 1000:.0f}K" if s >= 1000 else str(s)           # lerobot's format_big_number
        t = t0 + dt.timedelta(seconds=seconds_per_window * (i + 1))
        head = f"INFO {t:%Y-%m-%d %H:%M:%S} ot_train.py:641 " if stamp else "ot_train.py:641 "
        if tqdm:                                                        # the bar shares the line; matches TQDM_RE
            head = f"Training: {100 * s // steps:2d}%|xxxx| {s - lag}/{steps} [00:00<00:00, 1.00step/s]" + head
        lines.append(f"{head}step:{label} smpl:{s * 8} ep:0 epch:0.00 loss:0.1 grdn:1.0 lr:1.0e-05 updt_s:{updt_s:.3f} data_s:0.010 smp/s:80")
    return "\n".join(lines) + "\n"


def test_labels_only_20000_steps_at_log_freq_100_regression(tmp_path):
    """The shipped SUMMARY.json of every 20k-30k step run carried step_rate 0.0: labels repeat, most window deltas were 0.
    At 3000 steps the 8 exact pre-1K windows outvote the label-transition windows and the test cannot see the collapse."""
    rate, _ = _rate(tmp_path, _synth(20000, 100, 10.0))                 # 10 steps/s; 200 records, 29 distinct labels, 171 repeat one
    assert rate is not None and abs(rate - 10.0) < 0.5, rate


def test_tqdm_counter_is_used_when_the_labels_alone_would_mislead(tmp_path):
    # log_freq 500: lerobot's format_big_number rounds 1500 -> 2K, 2500 -> 2K, 3500 -> 4K, so distinct-label windows alternate
    # 500 / 1500 steps and labels alone give 2x; the per-step timers are deliberately wrong (updt_s 0.5)
    rate, _ = _rate(tmp_path, _synth(20000, 500, 50.0, tqdm=True, updt_s=0.5))
    assert abs(rate - 10.0) < 0.05, rate


def test_tqdm_counter_one_step_behind_the_label_still_rounds_to_it(tmp_path):
    # tqdm mininterval can leave the counter a step behind; log_freq 100
    rate, _ = _rate(tmp_path, _synth(20000, 100, 10.0, tqdm=True, lag=1))
    assert abs(rate - 10.0) < 0.05, rate


def test_smoke_log_200_steps_at_log_freq_50(tmp_path):
    rate, last = _rate(tmp_path, _synth(200, 50, 10.0))                  # 5 steps/s; first (warm-up) window skipped
    assert abs(rate - 5.0) < 0.01 and last["smp_per_s"] == 80.0


def test_no_timestamps_falls_back_to_the_per_step_timers(tmp_path):
    rate, _ = _rate(tmp_path, _synth(1000, 100, 10.0, stamp=False))
    assert abs(rate - 10.0) < 0.01                                      # 1 / (0.090 + 0.010)


def test_exact_step_trusts_the_counter_only_when_it_rounds_to_the_label():
    line = "Training:  53%|█████▎    | 10600/20000 [40:57<31:02,  5.05step/s]INFO 2026-09-11 15:54:00 ot_train.py:641 step:11K"
    assert summary.exact_step(line, 11000) == 10600
    assert summary.exact_step(line, 20000) == 20000                     # a counter from another record: keep the label
    assert summary.exact_step("INFO 2026-09-11 15:54:00 ot_train.py:641 step:11K", 11000) == 11000
    assert summary.exact_step("Training:  25%|██▌       | 50/200 [02:38<00:24,  6.13step/s]INFO 2026-09-10 13:13:49 step:50", 50) == 50
