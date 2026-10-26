import csv

from eval.analyze import accuracy_report, analyze, parameter_sweep, performance_report


def rows():
    return [
        {
            "timestamp": "0", "happy": "0.9", "sad": "0", "angry": "0", "neutral": "0.1", "surprised": "0",
            "raw_argmax": "happy", "smoothed_mood": "happy", "stable_mood": "happy", "switch_event": "1",
            "inference_ms": "10", "fps": "8", "smoothing_enabled": "1", "ground_truth": "happy",
        },
        {
            "timestamp": "60", "happy": "0", "sad": "0.8", "angry": "0", "neutral": "0.2", "surprised": "0",
            "raw_argmax": "sad", "smoothed_mood": "sad", "stable_mood": "sad", "switch_event": "1",
            "inference_ms": "20", "fps": "7", "smoothing_enabled": "0", "ground_truth": "sad",
        },
    ]


def test_accuracy_and_performance_reports():
    assert accuracy_report(rows())["accuracy"] == 1.0
    performance = performance_report(rows())
    assert performance["mean_inference_ms"] == 15
    assert performance["mean_fps"] == 7.5


def test_parameter_sweep_replays_raw_probabilities():
    result = parameter_sweep(rows(), windows=(1,), thresholds=(0.5,), dwell=(0,))

    assert result == [{"window": 1, "threshold": 0.5, "dwell": 0, "switches": 2}]


def test_analyze_writes_summary_and_sweep(tmp_path):
    log_path = tmp_path / "session.csv"
    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows()[0].keys())
        writer.writeheader()
        writer.writerows(rows())

    output = tmp_path / "results"
    analyze(log_path, output)

    assert (output / "summary.json").is_file()
    assert (output / "parameter_sweep.csv").is_file()


if __name__ == "__main__":
    from tempfile import TemporaryDirectory
    from pathlib import Path

    with TemporaryDirectory() as directory:
        root = Path(directory)
        test_accuracy_and_performance_reports()
        test_parameter_sweep_replays_raw_probabilities()
        test_analyze_writes_summary_and_sweep(root)
    print("Phase 7 evaluation tests passed.")