import json
import subprocess
import sys


def test_cli_writes_synthetic_report(tmp_path) -> None:
    path = tmp_path / "nested" / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "model_route",
            "--rounds",
            "1",
            "--personalize",
            "--report",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Synthetic offline experiment" in result.stdout
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["config"]["personalize"] is True
    assert data["summaries"]["router"]["decisions"] == 18
    assert len(data["decisions"]) == 90
