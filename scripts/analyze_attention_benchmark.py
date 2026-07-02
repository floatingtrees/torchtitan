#!/usr/bin/env python3
"""Summarize a multi-rank TorchTitan attention benchmark."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


SCALAR_TAGS = (
    "loss_metrics/global_avg_loss",
    "grad_norm",
    "memory/max_reserved(GiB)",
    "memory/num_alloc_retries",
    "memory/num_ooms",
)


def load_steps(path: str) -> tuple[int, dict[int, float]]:
    rank = None
    steps = {}
    with open(path) as handle:
        for line in handle:
            event = json.loads(line)
            if rank is None:
                rank = int(event["global_rank"])
            if event.get("log_type_name") == "step_end":
                steps[int(event["step"])] = float(event["value"])
    assert rank is not None
    return rank, steps


def load_scalars(path: str) -> dict[str, list[float]]:
    events = EventAccumulator(path)
    events.Reload()
    return {
        tag: [float(event.value) for event in events.Scalars(tag)]
        for tag in SCALAR_TAGS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()

    structured_paths = sorted(
        glob.glob(os.path.join(args.run_dir, "structured_logs", "*.jsonl"))
    )
    event_paths = sorted(
        glob.glob(
            os.path.join(
                args.run_dir,
                "tb",
                "*",
                "rank_*",
                "events.out.tfevents.*",
            )
        )
    )
    assert structured_paths
    assert len(structured_paths) == len(event_paths)

    by_rank = {}
    for path in structured_paths:
        rank, steps = load_steps(path)
        by_rank[rank] = steps
    num_steps = min(len(steps) for steps in by_rank.values())
    assert num_steps >= 10

    critical = {
        step: max(by_rank[rank][step] for rank in by_rank)
        for step in range(1, num_steps + 1)
    }
    steady_start = 11
    steady = [critical[step] for step in range(steady_start, num_steps + 1)]
    steady_median = statistics.median(steady)
    outlier_steps = {
        step
        for step in range(steady_start, num_steps + 1)
        if critical[step] > 5 * steady_median
    }
    clean = [
        critical[step]
        for step in range(steady_start, num_steps + 1)
        if step not in outlier_steps
    ]

    scalar_rows = [load_scalars(path) for path in event_paths]
    losses = scalar_rows[0]["loss_metrics/global_avg_loss"]
    grad_norms = scalar_rows[0]["grad_norm"]
    assert len(losses) == len(grad_norms) == num_steps

    wall_path = os.path.join(args.run_dir, "wall_time.txt")
    if os.path.exists(wall_path):
        print(open(wall_path).read().strip())
    print("critical_step_sum_seconds", sum(critical.values()) / 1000)
    print("cold_step1_seconds", critical[1] / 1000)
    print("steady_mean_ms", statistics.fmean(steady))
    print("steady_median_ms", steady_median)
    print("steady_p95_ms", float(np.percentile(steady, 95)))
    print(
        "outlier_steps_ms",
        {step: critical[step] for step in sorted(outlier_steps)},
    )
    print("outlier_clean_mean_ms", statistics.fmean(clean))
    print("outlier_clean_p95_ms", float(np.percentile(clean, 95)))
    print(
        "peak_reserved_gib",
        max(max(row["memory/max_reserved(GiB)"]) for row in scalar_rows),
    )
    print("loss_first", losses[0])
    print("loss_final", losses[-1])
    print("loss_first10_mean", statistics.fmean(losses[:10]))
    print("loss_last10_mean", statistics.fmean(losses[-10:]))
    print(
        "loss_all_finite",
        all(
            math.isfinite(value)
            for row in scalar_rows
            for value in row["loss_metrics/global_avg_loss"]
        ),
    )
    print(
        "loss_identical_across_ranks",
        all(
            row["loss_metrics/global_avg_loss"] == losses
            for row in scalar_rows[1:]
        ),
    )
    print("grad_norm_max", max(grad_norms))
    print(
        "grad_norm_all_finite",
        all(
            math.isfinite(value)
            for row in scalar_rows
            for value in row["grad_norm"]
        ),
    )
    print(
        "grad_norm_identical_across_ranks",
        all(row["grad_norm"] == grad_norms for row in scalar_rows[1:]),
    )
    print(
        "max_allocator_retries",
        max(max(row["memory/num_alloc_retries"]) for row in scalar_rows),
    )
    print(
        "max_ooms",
        max(max(row["memory/num_ooms"]) for row in scalar_rows),
    )


if __name__ == "__main__":
    main()
