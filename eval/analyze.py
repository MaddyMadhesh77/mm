"""Analyze MoodTune CSV logs for Phase 7 evaluation.

Example:
    python -m eval.analyze logs/session.csv --output eval-results
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from moodtune.config import MOODS
from moodtune.decision import DecisionEngine


def load_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def accuracy_report(rows: Iterable[Mapping[str, str]], prediction: str = "stable_mood") -> dict:
    pairs = [
        (row.get("ground_truth", "").strip().lower(), row.get(prediction, "").strip().lower())
        for row in rows
        if row.get("ground_truth", "").strip().lower() in MOODS
        and row.get(prediction, "").strip().lower() in MOODS
    ]
    matrix = {actual: {mood: 0 for mood in MOODS} for actual in MOODS}
    for actual, predicted in pairs:
        matrix[actual][predicted] += 1
    total = len(pairs)
    correct = sum(matrix[mood][mood] for mood in MOODS)
    per_class = {
        mood: matrix[mood][mood] / sum(matrix[mood].values()) if sum(matrix[mood].values()) else 0.0
        for mood in MOODS
    }
    return {"samples": total, "accuracy": correct / total if total else 0.0, "per_class": per_class, "matrix": matrix}


def performance_report(rows: Iterable[Mapping[str, str]]) -> dict:
    rows = list(rows)
    inference = [float(row["inference_ms"]) for row in rows if row.get("inference_ms", "")]
    fps = [float(row["fps"]) for row in rows if row.get("fps", "")]
    cpu = [float(row["cpu_percent"]) for row in rows if row.get("cpu_percent", "")]
    return {
        "samples": len(inference),
        "mean_inference_ms": statistics.fmean(inference) if inference else 0.0,
        "p95_inference_ms": _percentile(inference, 0.95),
        "mean_fps": statistics.fmean(fps) if fps else 0.0,
        "mean_cpu_percent": statistics.fmean(cpu) if cpu else None,
    }


def switch_report(rows: Iterable[Mapping[str, str]]) -> dict[str, float]:
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("smoothing_enabled", "1")].append(row)
    result = {}
    for enabled, group in grouped.items():
        if not group:
            continue
        timestamps = [float(row["timestamp"]) for row in group if row.get("timestamp", "")]
        duration_minutes = (max(timestamps) - min(timestamps)) / 60 if len(timestamps) > 1 else 0.0
        switches = sum(int(row.get("switch_event", "0")) for row in group)
        result["on" if enabled in {"1", "true", "True"} else "off"] = switches / duration_minutes if duration_minutes else float(switches)
    return result


def latency_report(rows: Iterable[Mapping[str, str]]) -> dict:
    """Measure cue-to-switch latency from ground-truth changes in the log.

    A cue is the first row where ``ground_truth`` changes to a new mood; its
    latency is the time until ``stable_mood`` first equals that mood.
    """
    records = [row for row in rows if row.get("timestamp", "")]
    latencies = []
    missed = 0
    previous_truth = None
    pending: tuple[str, float] | None = None
    for row in records:
        truth = row.get("ground_truth", "").strip().lower()
        timestamp = float(row["timestamp"])
        if truth in MOODS and truth != previous_truth:
            if pending:
                missed += 1
            pending = (truth, timestamp)
            previous_truth = truth
        if pending and row.get("stable_mood", "").strip().lower() == pending[0]:
            latencies.append(timestamp - pending[1])
            pending = None
    if pending:
        missed += 1
    return {
        "cues": len(latencies) + missed,
        "reached": len(latencies),
        "missed": missed,
        "mean_latency_seconds": statistics.fmean(latencies) if latencies else None,
        "median_latency_seconds": statistics.median(latencies) if latencies else None,
        "fade_seconds": 2.0,
    }


def parameter_sweep(rows: Iterable[Mapping[str, str]], windows=(1.0, 2.5, 5.0), thresholds=(0.4, 0.5, 0.6), dwell=(0.0, 15.0)) -> list[dict]:
    """Replay raw probabilities through DecisionEngine for offline tuning."""
    records = list(rows)
    results = []
    for window in windows:
        for threshold in thresholds:
            for dwell_seconds in dwell:
                engine = DecisionEngine(window, threshold, dwell_seconds)
                switches = 0
                for row in records:
                    probabilities = {mood: float(row.get(mood, 0.0) or 0.0) for mood in MOODS}
                    _, _, changed = engine.update(probabilities, float(row["timestamp"]))
                    switches += int(changed)
                results.append({"window": window, "threshold": threshold, "dwell": dwell_seconds, "switches": switches})
    return results


def write_plots(rows: list[dict[str, str]], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    report = accuracy_report(rows)
    matrix = [[report["matrix"][actual][predicted] for predicted in MOODS] for actual in MOODS]
    figure, axis = plt.subplots()
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks(range(len(MOODS)), MOODS)
    axis.set_yticks(range(len(MOODS)), MOODS)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Ground truth")
    figure.tight_layout()
    figure.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(figure)

    rates = switch_report(rows)
    figure, axis = plt.subplots()
    axis.bar(list(rates), [rates[key] for key in rates], color=["#5271ff", "#d62828"])
    axis.set_ylabel("Switches per minute")
    figure.tight_layout()
    figure.savefig(output / "switches_per_minute.png", dpi=160)
    plt.close(figure)


def analyze(path: str | Path, output: str | Path) -> dict:
    rows = load_rows(path)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    reports = {
        "accuracy": accuracy_report(rows),
        "accuracy_raw_frames": accuracy_report(rows, "raw_argmax"),
        "performance": performance_report(rows),
        "switches_per_minute": switch_report(rows),
        "latency": latency_report(rows),
    }
    (output_path / "summary.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    with (output_path / "parameter_sweep.csv").open("w", newline="", encoding="utf-8") as file:
        sweep = parameter_sweep(rows)
        writer = csv.DictWriter(file, fieldnames=sweep[0].keys() if sweep else ["window", "threshold", "dwell", "switches"])
        writer.writeheader()
        writer.writerows(sweep)
    write_plots(rows, output_path)
    (output_path / "report_tables.md").write_text(markdown_tables(reports), encoding="utf-8")
    return reports


def markdown_tables(reports: dict) -> str:
    """Render the measured values as tables ready to paste into Chapter 5."""
    def fmt(value, suffix=""):
        return "—" if value is None else f"{value:.2f}{suffix}"

    accuracy, raw = reports["accuracy"], reports["accuracy_raw_frames"]
    lines = ["## Table 5.1: Per-class accuracy", "", "| Mood | Smoothed | Raw frames |", "|---|---|---|"]
    for mood in MOODS:
        lines.append(f"| {mood.title()} | {accuracy['per_class'][mood]:.1%} | {raw['per_class'][mood]:.1%} |")
    lines.append(f"| **Overall** | **{accuracy['accuracy']:.1%}** | **{raw['accuracy']:.1%}** |")
    lines += ["", "Confusion matrix (rows = requested, columns = predicted):", "",
              "| | " + " | ".join(m.title() for m in MOODS) + " |", "|---" * (len(MOODS) + 1) + "|"]
    for actual in MOODS:
        lines.append(f"| {actual.title()} | " + " | ".join(str(accuracy["matrix"][actual][p]) for p in MOODS) + " |")

    performance, latency, switches = reports["performance"], reports["latency"], reports["switches_per_minute"]
    lines += ["", "## Table 5.2: Performance and stability", "", "| Metric | Value |", "|---|---|",
              f"| Mean inference time | {fmt(performance['mean_inference_ms'], ' ms')} |",
              f"| 95th percentile inference time | {fmt(performance['p95_inference_ms'], ' ms')} |",
              f"| Mean loop FPS | {fmt(performance['mean_fps'])} |",
              f"| Mean CPU use | {fmt(performance['mean_cpu_percent'], '%')} |",
              f"| Mean cue-to-switch latency | {fmt(latency['mean_latency_seconds'], ' s')} (+{latency['fade_seconds']:.0f} s fade) |",
              f"| Cues reached / total | {latency['reached']} / {latency['cues']} |",
              f"| Switches per minute, smoothing ON | {fmt(switches.get('on'))} |",
              f"| Switches per minute, smoothing OFF | {fmt(switches.get('off'))} |", ""]
    return "\n".join(lines)


def _differences(values: list[float]) -> list[float]:
    return [later - earlier for earlier, later in zip(values, values[1:])]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, int((len(values) - 1) * percentile))
    return values[index]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log")
    parser.add_argument("--output", default="eval-results")
    arguments = parser.parse_args()
    print(json.dumps(analyze(arguments.log, arguments.output), indent=2))