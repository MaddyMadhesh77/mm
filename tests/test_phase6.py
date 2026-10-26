import csv

from moodtune.logger import FIELDNAMES, CsvLogger


def test_logger_writes_required_evaluation_row(tmp_path):
    path = tmp_path / "session.csv"
    with CsvLogger(path) as logger:
        logger.log(
            timestamp=1.5,
            probabilities={"happy": 0.8, "neutral": 0.2},
            smoothed_mood="happy",
            stable_mood="happy",
            switch_event=True,
            inference_ms=12.4,
            fps=7.8,
            smoothing_enabled=True,
        )

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert list(rows[0]) == list(FIELDNAMES)
    assert rows[0]["raw_argmax"] == "happy"
    assert rows[0]["switch_event"] == "1"
    assert rows[0]["smoothing_enabled"] == "1"


if __name__ == "__main__":
    from tempfile import TemporaryDirectory
    from pathlib import Path

    with TemporaryDirectory() as directory:
        test_logger_writes_required_evaluation_row(Path(directory))
    print("Phase 6 logging tests passed.")