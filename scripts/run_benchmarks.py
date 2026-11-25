"""Reproducible Offline Latency Benchmark Suite for Onmyoji Tactical Helper.

Measures P50 / P95 / P99 / Mean execution latency (in milliseconds) across core CPU
decision, spatial tracking, geometric CV, and mixed-DPI coordinate-mapping stages,
writing structured results to `docs/BENCHMARKS.json`.

Usage:
    python scripts/run_benchmarks.py
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from onmyoji_helper.automation.coordinate_mapper import (  # noqa: E402
    CoordinateMapper,
    project_cropped_to_physical_client,
)
from onmyoji_helper.decision.draft_engine import build_database, get_recommendation  # noqa: E402
from onmyoji_helper.domain.models import CropOffset  # noqa: E402
from onmyoji_helper.domain.public_demo_data import get_public_demo_bundle  # noqa: E402
from onmyoji_helper.domain.shikigami_repository import ShikigamiRepository  # noqa: E402
from onmyoji_helper.vision.draft_ocr_scanner import get_downsampled_blur  # noqa: E402
from onmyoji_helper.vision.spatial_tracker import StateParser  # noqa: E402
from onmyoji_helper.vision.turn_detector import TurnDetector  # noqa: E402


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = (len(sorted_vals) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _measure(fn: Callable[[], Any], iterations: int = 1000, warmup: int = 80) -> dict[str, float | int]:
    for _ in range(warmup):
        fn()

    samples_ms: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        fn()
        elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        samples_ms.append(elapsed_ms)

    samples_ms.sort()
    return {
        "iterations": iterations,
        "p50_ms": round(_percentile(samples_ms, 0.50), 4),
        "p95_ms": round(_percentile(samples_ms, 0.95), 4),
        "p99_ms": round(_percentile(samples_ms, 0.99), 4),
        "mean_ms": round(statistics.fmean(samples_ms), 4),
    }


def _build_synthetic_turn_frame(detector: TurnDetector, w: int = 1280, h: int = 720) -> np.ndarray:
    frame = np.full((h, w, 3), (18, 20, 24), dtype=np.uint8)
    for slot_key, lit in (("slot_1", True), ("slot_2", False), ("slot_3", True), ("slot_4", True)):
        rx1, ry1, rx2, ry2 = detector.PHYSICAL_SLOTS[slot_key]
        cx = int(round(w * (rx1 + rx2) * 0.5))
        cy = int(round(h * (ry1 + ry2) * 0.5))
        if lit:
            cv2.circle(frame, (cx, cy), 27, (38, 178, 245), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 28, (235, 245, 255), thickness=2, lineType=cv2.LINE_AA)
        else:
            cv2.circle(frame, (cx, cy), 27, (54, 54, 54), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 28, (158, 158, 158), thickness=2, lineType=cv2.LINE_AA)
    return frame


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)

    systems, dict_map, multi_shorts, tags_db = get_public_demo_bundle()
    repo = ShikigamiRepository(tags_db=tags_db, full_to_short_map=dict_map, multi_char_shorts=multi_shorts)
    first_sys = next(iter(systems.values()))
    db = build_database(first_sys["raw_data"], repo.sorted_multi_shorts)

    # 1. XMind Counter-Tree + Viability Decay + Mirror Detection
    def bench_xmind() -> None:
        get_recommendation(
            db=db,
            current_ban="Ban封",
            opp_draft=["葛", "神", "石"],
            my_draft=["離", "百羽"],
            my_ban="禪",
        )

    # 2. ShikigamiRepository LRU Tag Lookup
    shiki_keys = ["閻", "玲", "神荒", "追月", "SP06", "葛", "帝", "連"]
    idx_counter = [0]

    def bench_repo_lookup() -> None:
        k = shiki_keys[idx_counter[0] & 7]
        idx_counter[0] += 1
        repo.get_tag_info(k)

    # 3. Noisy 2D Partial Affine + EMA Occlusion Spatial Tracker
    rng = random.Random(2026)
    parser = StateParser()
    w, h = 1144, 640
    my_draft = ["葛", "帝", "骷", "閻", "連"]
    opp_draft = ["玲", "SP桃", "葛", "酒吞", "季"]
    init_boxes: list[dict[str, float | int]] = [
        {"cls": 0, "cx": w * r, "cy": h * 0.30} for r in (0.48, 0.57, 0.66, 0.75, 0.84)
    ] + [
        {"cls": 1, "cx": w * r, "cy": h * 0.62} for r in (0.130, 0.315, 0.495, 0.665, 0.835)
    ]
    parser.parse_frame(my_draft, opp_draft, init_boxes, w, h, now=10.0)
    step_clock = [10.25]

    def bench_noisy_affine_tracker() -> None:
        step_clock[0] += 0.25
        # Keep within 3.5s grace by refreshing full visibility every 8 frames
        occlude = (int(step_clock[0] * 4) % 8) != 0
        boxes: list[dict[str, float | int]] = []
        for idx, r in enumerate((0.48, 0.57, 0.66, 0.75, 0.84)):
            if occlude and idx == 3:
                continue
            boxes.append(
                {
                    "cls": 0,
                    "cx": w * r + 12.0 + max(-4.0, min(4.0, rng.gauss(0.0, 2.0))),
                    "cy": h * 0.30 - 6.0 + max(-4.0, min(4.0, rng.gauss(0.0, 2.0))),
                }
            )
        for r in (0.130, 0.315, 0.495, 0.665, 0.835):
            boxes.append(
                {
                    "cls": 1,
                    "cx": w * r + max(-4.0, min(4.0, rng.gauss(0.0, 2.0))),
                    "cy": h * 0.62 + max(-4.0, min(4.0, rng.gauss(0.0, 2.0))),
                }
            )
        parser.parse_frame(my_draft, opp_draft, boxes, w, h, now=step_clock[0])

    # 4. Mixed-DPI (125%) + Letterbox Physical Coordinate Projection & Jitter Clamp
    crop = CropOffset(x0=0, y0=72, client_w=1920, client_h=1200)
    bounds = (0, 0, 1920, 1200)

    def bench_dpi_projection() -> None:
        px, py = project_cropped_to_physical_client(858.0, 192.0, 1536.0, 864.0, 1920.0, 1200.0, crop=crop)
        CoordinateMapper.apply_gaussian_jitter(int(px), int(py), 6.0, bounds)

    # 5. Draft ROI 120x80 Gaussian Motion-Diff Gate
    roi_a = np.random.default_rng(7).integers(0, 255, size=(345, 700), dtype=np.uint8)
    blur_prev = get_downsampled_blur(roi_a)
    assert blur_prev is not None

    def bench_motion_diff_gate() -> None:
        curr = get_downsampled_blur(roi_a)
        assert curr is not None
        _ = cv2.mean(cv2.absdiff(curr, blur_prev))[0]

    # 6. Full TurnDetector Geometric Pass (CLAHE + HoughCircles + 16-pt Sobel Ring + 4-slot HSV)
    detector = TurnDetector()
    turn_frame = _build_synthetic_turn_frame(detector, 1280, 720)
    detector.is_turn_confirmed = True

    def bench_turn_detector_full() -> None:
        detector.reset_cache()
        detector.is_turn_confirmed = True
        detector.locked_skill_mode = None
        detector.detect_available_skills(turn_frame, frame_seq=1)

    results = {
        "metadata": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "opencv_version": cv2.__version__,
            "numpy_version": np.__version__,
            "command": "python scripts/run_benchmarks.py",
        },
        "benchmarks": {
            "xmind_counter_tree_scoring": _measure(bench_xmind, iterations=2000),
            "shikigami_repo_lru_lookup": _measure(bench_repo_lookup, iterations=5000),
            "noisy_2d_affine_spatial_tracker": _measure(bench_noisy_affine_tracker, iterations=1000),
            "mixed_dpi_coordinate_projection": _measure(bench_dpi_projection, iterations=5000),
            "draft_roi_motion_diff_gate": _measure(bench_motion_diff_gate, iterations=2000),
            "turn_detector_sobel_hsv_full_pass": _measure(bench_turn_detector_full, iterations=500),
        },
    }

    out_path = PROJECT_ROOT / "docs" / "BENCHMARKS.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"Saved benchmark results to {out_path}\n")
    print("| Stage | P50 (ms) | P95 (ms) | P99 (ms) | Mean (ms) |")
    print("| :--- | :---: | :---: | :---: | :---: |")
    for name, m in results["benchmarks"].items():
        print(
            f"| `{name}` | `{m['p50_ms']:.3f}` | `{m['p95_ms']:.3f}` | "
            f"`{m['p99_ms']:.3f}` | `{m['mean_ms']:.3f}` |"
        )


if __name__ == "__main__":
    main()
