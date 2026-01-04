from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import mlflow


def load_last_run_id(path: Path) -> str:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    run_id = data.get("run_id")
    if not run_id:
        raise ValueError(f"No run_id found in {path}")
    return str(run_id)


def export_model(run_id: str, artifact_subpath: str, out_dir: Path) -> None:
    """
    Downloads the MLflow model artifacts from a run into out_dir.
    This produces an MLflow model folder containing MLmodel + conda.yaml/python_env.yaml + model.pkl etc.
    """
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    tmp = Path(".tmp_export") / artifact_subpath
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.parent.mkdir(parents=True, exist_ok=True)

    # Download the artifact directory for the model
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=artifact_subpath,
        dst_path=str(tmp),
    )

    # local_path points to the downloaded directory
    src = Path(local_path)

    if out_dir.exists():
        shutil.rmtree(out_dir)

    shutil.copytree(src, out_dir)
    print(f"Exported {artifact_subpath} from run {run_id} -> {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Export latest MLflow models into repo for Streamlit Cloud demo."
    )
    p.add_argument(
        "--run-json",
        default="reports/artifacts/last_run.json",
        help="Path to last_run.json produced by training.",
    )
    p.add_argument(
        "--baseline-artifact",
        default="baseline_model",
        help="MLflow artifact path for baseline model.",
    )
    p.add_argument(
        "--mitigated-artifact",
        default="mitigated_model",
        help="MLflow artifact path for mitigated model.",
    )
    p.add_argument(
        "--out-root",
        default="models",
        help="Output root directory to store demo models.",
    )
    args = p.parse_args()

    run_json = Path(args.run_json)
    run_id = load_last_run_id(run_json)

    out_root = Path(args.out_root)
    export_model(run_id, args.baseline_artifact, out_root / "demo_baseline")
    export_model(run_id, args.mitigated_artifact, out_root / "demo_mitigated")


if __name__ == "__main__":
    main()
