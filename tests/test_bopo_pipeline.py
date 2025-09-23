import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("torch")


def test_bopo_pipeline_example(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "bopo_multiobj_pipeline.py"
    assays_path = repo_root / "data/examples/bopo_example_assays.tsv"
    candidates_path = repo_root / "data/examples/bopo_example_candidates.tsv"
    out_selected = tmp_path / "selected.tsv"

    cmd = [
        sys.executable,
        str(script),
        "--assays_tsv",
        str(assays_path),
        "--objective_cols",
        "score",
        "--candidate_tsv",
        str(candidates_path),
        "--m",
        "3",
        "--epochs",
        "5",
        "--batch_size",
        "4",
        "--hidden_dim",
        "16",
        "--lr",
        "5e-3",
        "--out_selected",
        str(out_selected),
        "--seed",
        "123",
    ]

    subprocess.run(cmd, check=True, cwd=repo_root)

    selected_df = pd.read_csv(out_selected, sep="\t")
    assert list(selected_df.columns) == ["rank", "mutations", "predicted_score", "diversity"]
    assert selected_df["rank"].tolist() == [1, 2, 3]
    assert selected_df["mutations"].notna().all()

    candidate_df = pd.read_csv(candidates_path, sep="\t")
    candidate_set = set(candidate_df["mutations"].tolist())
    assert set(selected_df["mutations"].tolist()).issubset(candidate_set)

    assert (selected_df["predicted_score"].astype(float).notna()).all()
    assert (selected_df["diversity"] >= 0).all()
