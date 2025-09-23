"""End-to-end BOPO-style preference pipeline with multi-objective selection.

This script implements the workflow described in the prompt:

* Loads assay results stored in a TSV file with ``batch`` and ``mutations`` columns
  alongside one or more objective columns.
* Builds within-batch preference pairs, optionally augmenting them with
  best-anchored comparisons and per-pair weights derived from the positive
  improvement across all objectives.
* Trains a small preference model (POCO/BOPO-style) using PyTorch.
* Scores candidate variants supplied explicitly or generated from deep mutational
  scanning (DMS) singles, then selects the next ``m`` variants on a Pareto front
  balancing predicted score and diversity measured as Hamming distance to anchor
  seeds and already-selected variants.

Example
-------

.. code-block:: bash

   # Train with a single objective and propose 24 variants from DMS singles
   python bopo_multiobj_pipeline.py \
       --assays_tsv assays.tsv \
       --objective_cols score \
       --dms_tsv dms.tsv \
       --top_n_mutations 25 \
       --max_mutations_per_variant 3 \
       --max_candidates 4000 \
       --m 24 \
       --epochs 50 \
       --lr 1e-3 \
       --out_selected selected_variants.tsv

   # Train with two objectives using an explicit candidate list
   python bopo_multiobj_pipeline.py \
       --assays_tsv assays.tsv \
       --objective_cols score,stress_score \
       --candidate_tsv candidates.tsv \
       --m 16 \
       --epochs 80 \
       --lr 5e-4 \
       --out_selected selected_variants.tsv
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


MutationTokens = List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assays_tsv", required=True, type=Path, help="Path to assay TSV")
    parser.add_argument(
        "--objective_cols",
        required=True,
        type=str,
        help="Comma-separated list of objective column names present in assays TSV.",
    )
    parser.add_argument(
        "--candidate_tsv",
        type=Path,
        default=None,
        help="Optional TSV with a 'mutations' column listing candidate variants.",
    )
    parser.add_argument(
        "--dms_tsv",
        type=Path,
        default=None,
        help=(
            "Optional DMS TSV with 'mutation' and optional 'score' columns used to "
            "build candidate combinations when --candidate_tsv is not supplied."
        ),
    )
    parser.add_argument(
        "--top_n_mutations",
        type=int,
        default=25,
        help="Number of top single mutations (by DMS score) used for combination generation.",
    )
    parser.add_argument(
        "--max_mutations_per_variant",
        type=int,
        default=3,
        help="Maximum number of mutations per generated candidate when using DMS input.",
    )
    parser.add_argument(
        "--max_candidates",
        type=int,
        default=4000,
        help="Upper bound on the number of generated candidates when using DMS input.",
    )
    parser.add_argument(
        "--anchors_file",
        type=Path,
        default=None,
        help="Optional file with one mutations string per line specifying anchor variants.",
    )
    parser.add_argument(
        "--anchor_pair_weight",
        type=float,
        default=2.0,
        help=(
            "Multiplicative weight applied to best-anchored pairs. Values >1.0 emphasise "
            "comparisons involving the batch-best variant."
        ),
    )
    parser.add_argument(
        "--m",
        type=int,
        required=True,
        help="Number of variants to select for the next batch.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs for the preference net.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Mini-batch size for preference training.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden size for the feed-forward preference network.",
    )
    parser.add_argument(
        "--weight_normalization",
        action="store_true",
        help="Normalize pair weights to have mean 1 (recommended when magnitudes vary).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device (cpu or cuda). Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for numpy, torch and Python RNGs.",
    )
    parser.add_argument(
        "--out_selected",
        required=True,
        type=Path,
        help="Output TSV path for the selected variants.",
    )
    parser.add_argument(
        "--out_pairs",
        type=Path,
        default=None,
        help="Optional TSV path to dump the generated preference pairs for auditing.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Increase logging verbosity.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_TOKEN_SPLIT_RE = re.compile(r"[;,]")
_POSITION_RE = re.compile(r"(\d+)")


def parse_mutation_string(mutation_str: str) -> MutationTokens:
    """Split a mutation string like ``"A23G;L54M"`` into canonical tokens."""

    if mutation_str is None:
        return []
    mutation_str = str(mutation_str).strip()
    if not mutation_str or mutation_str.upper() in {"WT", "CTRL", "CONTROL"}:
        return []
    tokens = [tok.strip() for tok in _TOKEN_SPLIT_RE.split(mutation_str) if tok.strip()]
    return tokens


def extract_position(token: str) -> Optional[int]:
    match = _POSITION_RE.search(token)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def canonicalize_tokens(tokens: MutationTokens) -> MutationTokens:
    def sort_key(tok: str) -> Tuple[int, str]:
        position = extract_position(tok)
        return (position if position is not None else math.inf, tok)

    return sorted(tokens, key=sort_key)


def canonical_mutation_string(mutation_str: str) -> str:
    tokens = canonicalize_tokens(parse_mutation_string(mutation_str))
    if not tokens:
        return "WT"
    return ";".join(tokens)


class MutationFeaturizer:
    """Simple binary featurizer mapping mutation tokens to a fixed vocabulary."""

    def __init__(self, vocabulary: Sequence[str]):
        self.vocabulary: List[str] = list(vocabulary)
        self.index: Dict[str, int] = {mut: i for i, mut in enumerate(self.vocabulary)}

    @property
    def dim(self) -> int:
        return len(self.vocabulary)

    def transform(self, mutation_strings: Sequence[str]) -> torch.Tensor:
        mat = np.zeros((len(mutation_strings), self.dim), dtype=np.float32)
        for row_idx, mutation_str in enumerate(mutation_strings):
            for token in parse_mutation_string(mutation_str):
                idx = self.index.get(token)
                if idx is None:
                    idx = self.index.get(canonical_mutation_string(token))
                if idx is not None:
                    mat[row_idx, idx] = 1.0
        return torch.from_numpy(mat)


def compute_scalar_score(row: pd.Series, objectives: Sequence[str]) -> float:
    return float(sum(float(row[col]) for col in objectives))


@dataclass
class PreferencePair:
    winner: str
    loser: str
    batch: str
    weight: float


def generate_preference_pairs(
    df: pd.DataFrame,
    objective_cols: Sequence[str],
    anchor_weight: float = 2.0,
    weight_normalization: bool = False,
) -> List[PreferencePair]:
    pairs: List[PreferencePair] = []
    for batch, group in df.groupby("batch"):
        group_records = group.to_dict("records")
        if not group_records:
            continue
        scores = [compute_scalar_score(pd.Series(rec), objective_cols) for rec in group_records]
        best_score = max(scores)
        best_indices = {i for i, score in enumerate(scores) if math.isclose(score, best_score)}
        for i in range(len(group_records)):
            for j in range(i + 1, len(group_records)):
                rec_i = group_records[i]
                rec_j = group_records[j]
                score_i = scores[i]
                score_j = scores[j]
                if math.isclose(score_i, score_j):
                    continue
                if score_i > score_j:
                    winner, loser = rec_i, rec_j
                    winner_idx = i
                    delta = np.maximum(
                        np.array([float(rec_i[col]) - float(rec_j[col]) for col in objective_cols]),
                        0.0,
                    )
                else:
                    winner, loser = rec_j, rec_i
                    winner_idx = j
                    delta = np.maximum(
                        np.array([float(rec_j[col]) - float(rec_i[col]) for col in objective_cols]),
                        0.0,
                    )
                weight = float(delta.sum())
                weight = max(weight, 1e-6)
                # Boost weight if this comparison involves the batch-best winner.
                if winner_idx in best_indices:
                    weight *= anchor_weight
                pairs.append(
                    PreferencePair(
                        winner=winner["mutations"],
                        loser=loser["mutations"],
                        batch=str(batch),
                        weight=weight,
                    )
                )

    if weight_normalization and pairs:
        weights = np.array([pair.weight for pair in pairs], dtype=np.float32)
        mean = float(weights.mean())
        if mean > 0:
            for pair in pairs:
                pair.weight = pair.weight / mean
    return pairs


def load_anchors(df: pd.DataFrame, anchors_file: Optional[Path]) -> Set[str]:
    anchors: Set[str] = set()
    if anchors_file and anchors_file.exists():
        anchors = {
            canonical_mutation_string(line.strip())
            for line in anchors_file.read_text().splitlines()
            if line.strip()
        }
    else:
        counts = df.groupby("mutations")["batch"].nunique()
        anchors = set(counts[counts >= 2].index.tolist())
    return anchors


def build_vocabulary(strings: Iterable[str]) -> List[str]:
    tokens: Set[str] = set()
    for mutation_str in strings:
        for token in parse_mutation_string(mutation_str):
            tokens.add(token)
    return sorted(tokens, key=lambda tok: (extract_position(tok) or math.inf, tok))


def _position_conflict(tokens: Sequence[str]) -> bool:
    seen: Set[int] = set()
    for tok in tokens:
        pos = extract_position(tok)
        if pos is None:
            continue
        if pos in seen:
            return True
        seen.add(pos)
    return False


def generate_candidates_from_dms(
    dms_path: Path,
    top_n: int,
    max_k: int,
    max_candidates: int,
) -> List[str]:
    sep = "\t" if dms_path.suffix in {".tsv", ".txt"} else ","
    dms_df = pd.read_csv(dms_path, sep=sep)
    if "mutation" not in dms_df.columns:
        raise ValueError("DMS TSV must contain a 'mutation' column")
    dms_df = dms_df.dropna(subset=["mutation"]).copy()
    if "score" in dms_df.columns:
        dms_df = dms_df.sort_values("score", ascending=False)
    mutation_series = dms_df["mutation"].astype(str)
    top_mutations = []
    for mut in mutation_series:
        canonical = canonical_mutation_string(mut)
        if canonical != "WT" and canonical not in top_mutations:
            top_mutations.append(canonical)
        if len(top_mutations) >= top_n:
            break
    score_map: Dict[str, float] = {}
    score_values = dms_df["score"].tolist() if "score" in dms_df.columns else [0.0] * len(dms_df)
    for mut, score in zip(mutation_series, score_values):
        canonical = canonical_mutation_string(mut)
        if canonical == "WT":
            continue
        try:
            score_float = float(score)
        except (TypeError, ValueError):
            score_float = 0.0
        score_map[canonical] = max(score_map.get(canonical, float("-inf")), score_float)

    candidates: List[Tuple[str, float]] = []
    for k in range(1, max_k + 1):
        for combo in itertools.combinations(top_mutations, k):
            if _position_conflict(combo):
                continue
            canonical_combo = canonicalize_tokens(list(combo))
            mut_str = ";".join(canonical_combo)
            combo_score = sum(score_map.get(token, 0.0) for token in canonical_combo)
            candidates.append((mut_str, combo_score))
    # Remove duplicates keeping the best combo score.
    best_scores: Dict[str, float] = {}
    for mut_str, combo_score in candidates:
        best_scores[mut_str] = max(best_scores.get(mut_str, float("-inf")), combo_score)
    ranked = sorted(best_scores.items(), key=lambda item: item[1], reverse=True)
    ranked = ranked[:max_candidates]
    return [mut_str for mut_str, _ in ranked]


class PreferenceNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x).squeeze(-1)


def preference_loss(
    model: PreferenceNet,
    x_win: torch.Tensor,
    x_lose: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    scores_win = model(x_win)
    scores_lose = model(x_lose)
    log_sigmoid = torch.logsigmoid(scores_win - scores_lose)
    if weights is not None:
        return -(weights * log_sigmoid).mean()
    return -log_sigmoid.mean()


def train_preference_model(
    model: PreferenceNet,
    optimizer: torch.optim.Optimizer,
    featurizer: MutationFeaturizer,
    pairs: Sequence[PreferencePair],
    epochs: int,
    batch_size: int,
    device: torch.device,
    verbose: bool = False,
) -> None:
    if not pairs:
        raise ValueError("No preference pairs available for training.")

    winner_features = featurizer.transform([pair.winner for pair in pairs]).to(device)
    loser_features = featurizer.transform([pair.loser for pair in pairs]).to(device)
    weights = torch.tensor([pair.weight for pair in pairs], dtype=torch.float32, device=device)

    num_pairs = winner_features.size(0)
    indices = torch.arange(num_pairs, device=device)

    model.to(device)
    model.train()
    for epoch in range(1, epochs + 1):
        perm = indices[torch.randperm(num_pairs)]
        for start in range(0, num_pairs, batch_size):
            batch_idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            loss = preference_loss(
                model,
                winner_features[batch_idx],
                loser_features[batch_idx],
                weights=weights[batch_idx],
            )
            loss.backward()
            optimizer.step()
        if verbose and epoch % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch:04d} | loss={loss.item():.4f}")


def hamming_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    return float(np.abs(vec_a - vec_b).sum())


def nondominated_sort(scores: np.ndarray, diversity: np.ndarray) -> List[List[int]]:
    population_size = len(scores)
    dominates: List[List[int]] = [[] for _ in range(population_size)]
    domination_counts = [0] * population_size
    fronts: List[List[int]] = [[]]

    for p in range(population_size):
        for q in range(population_size):
            if p == q:
                continue
            if (scores[p] >= scores[q] and diversity[p] >= diversity[q]) and (
                scores[p] > scores[q] or diversity[p] > diversity[q]
            ):
                dominates[p].append(q)
            elif (scores[q] >= scores[p] and diversity[q] >= diversity[p]) and (
                scores[q] > scores[p] or diversity[q] > diversity[p]
            ):
                domination_counts[p] += 1
        if domination_counts[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front: List[int] = []
        for p in fronts[i]:
            for q in dominates[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    if not fronts[-1]:
        fronts.pop()
    return fronts


def crowding_distance(scores: np.ndarray, diversity: np.ndarray) -> np.ndarray:
    n = len(scores)
    if n == 0:
        return np.array([])
    distances = np.zeros(n, dtype=np.float32)
    if n <= 2:
        distances.fill(np.inf)
        return distances

    order_score = np.argsort(scores)
    order_div = np.argsort(diversity)
    distances[order_score[0]] = distances[order_score[-1]] = np.inf
    distances[order_div[0]] = distances[order_div[-1]] = np.inf

    score_range = scores[order_score[-1]] - scores[order_score[0]]
    div_range = diversity[order_div[-1]] - diversity[order_div[0]]
    score_range = score_range if score_range != 0 else 1.0
    div_range = div_range if div_range != 0 else 1.0

    for i in range(1, n - 1):
        distances[order_score[i]] += (
            scores[order_score[i + 1]] - scores[order_score[i - 1]]
        ) / score_range
        distances[order_div[i]] += (
            diversity[order_div[i + 1]] - diversity[order_div[i - 1]]
        ) / div_range
    return distances


def pareto_selection(
    candidate_strings: Sequence[str],
    candidate_scores: np.ndarray,
    feature_matrix: np.ndarray,
    seed_features: List[np.ndarray],
    m: int,
) -> Tuple[List[int], List[float], List[float]]:
    selected_indices: List[int] = []
    selected_scores: List[float] = []
    selected_diversity: List[float] = []

    remaining: Set[int] = set(range(len(candidate_strings)))
    seed_vectors = list(seed_features)

    while remaining and len(selected_indices) < m:
        rem_indices = sorted(list(remaining))
        rem_scores = candidate_scores[rem_indices]
        rem_diversity = []
        for idx in rem_indices:
            vec = feature_matrix[idx]
            if seed_vectors:
                distances = [hamming_distance(vec, seed_vec) for seed_vec in seed_vectors]
                rem_diversity.append(min(distances))
            else:
                rem_diversity.append(float(np.count_nonzero(vec)))
        rem_diversity_arr = np.array(rem_diversity, dtype=np.float32)
        fronts = nondominated_sort(rem_scores, rem_diversity_arr)
        if not fronts:
            break
        front = fronts[0]
        front_scores = rem_scores[front]
        front_div = rem_diversity_arr[front]
        distances = crowding_distance(front_scores, front_div)
        best_idx_within_front = int(np.argmax(distances + 1e-6 * front_scores))
        chosen_idx = rem_indices[front[best_idx_within_front]]
        selected_indices.append(chosen_idx)
        selected_scores.append(float(candidate_scores[chosen_idx]))
        selected_diversity.append(float(rem_diversity_arr[front[best_idx_within_front]]))
        seed_vectors.append(feature_matrix[chosen_idx])
        remaining.remove(chosen_idx)

    return selected_indices, selected_scores, selected_diversity


def ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    objective_cols = [col.strip() for col in args.objective_cols.split(",") if col.strip()]
    if not objective_cols:
        raise ValueError("At least one objective column must be specified via --objective_cols")

    if not args.assays_tsv.exists():
        raise FileNotFoundError(f"Assay file not found: {args.assays_tsv}")
    assays_df = pd.read_csv(args.assays_tsv, sep="\t")
    ensure_columns(assays_df, ["batch", "mutations", *objective_cols])
    assays_df = assays_df.copy()
    assays_df["mutations"] = assays_df["mutations"].map(canonical_mutation_string)
    for col in objective_cols:
        assays_df[col] = pd.to_numeric(assays_df[col], errors="coerce")
    missing_mask = assays_df[objective_cols].isna().any(axis=1)
    if missing_mask.any():
        dropped = int(missing_mask.sum())
        assays_df = assays_df.loc[~missing_mask].copy()
        if args.verbose:
            print(f"Dropped {dropped} assay rows with missing objective values")
    if assays_df.empty():
        raise ValueError("No assay rows remaining after cleaning; cannot proceed.")
    assays_df = assays_df.sort_values("batch").reset_index(drop=True)

    anchors = load_anchors(assays_df, args.anchors_file)
    if args.verbose:
        print(f"Identified {len(anchors)} anchor variants")

    pairs = generate_preference_pairs(
        assays_df,
        objective_cols,
        anchor_weight=args.anchor_pair_weight,
        weight_normalization=args.weight_normalization,
    )
    if args.verbose:
        print(f"Generated {len(pairs)} preference pairs")

    if args.out_pairs:
        pair_rows = [
            {
                "winner": pair.winner,
                "loser": pair.loser,
                "batch": pair.batch,
                "weight": pair.weight,
            }
            for pair in pairs
        ]
        out_df = pd.DataFrame(pair_rows)
        out_df.to_csv(args.out_pairs, index=False, sep="\t")
        if args.verbose:
            print(f"Wrote preference pairs to {args.out_pairs}")

    if args.candidate_tsv:
        if not args.candidate_tsv.exists():
            raise FileNotFoundError(f"Candidate file not found: {args.candidate_tsv}")
        sep = "\t" if args.candidate_tsv.suffix in {".tsv", ".txt"} else ","
        candidates_df = pd.read_csv(args.candidate_tsv, sep=sep)
        ensure_columns(candidates_df, ["mutations"])
        candidate_strings = [canonical_mutation_string(s) for s in candidates_df["mutations"].tolist()]
    else:
        if not args.dms_tsv:
            raise ValueError("Either --candidate_tsv or --dms_tsv must be supplied.")
        if not args.dms_tsv.exists():
            raise FileNotFoundError(f"DMS file not found: {args.dms_tsv}")
        candidate_strings = generate_candidates_from_dms(
            args.dms_tsv,
            top_n=args.top_n_mutations,
            max_k=args.max_mutations_per_variant,
            max_candidates=args.max_candidates,
        )
        if args.verbose:
            print(f"Generated {len(candidate_strings)} candidates from DMS scores")

    # Merge anchors into candidate set for deduplication without altering selection pool.
    candidate_strings = sorted(set(candidate_strings))
    if len(candidate_strings) < args.m:
        raise ValueError(
            f"Need at least m={args.m} unique candidates, but only {len(candidate_strings)} were provided."
        )

    vocabulary = build_vocabulary(
        list(assays_df["mutations"].tolist()) + candidate_strings + list(anchors)
    )
    featurizer = MutationFeaturizer(vocabulary)

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if args.verbose:
        print(f"Using device: {device}")

    model = PreferenceNet(featurizer.dim, hidden_dim=args.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_preference_model(
        model,
        optimizer,
        featurizer,
        pairs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        verbose=args.verbose,
    )

    candidate_tensor = featurizer.transform(candidate_strings)
    model.eval()
    with torch.no_grad():
        candidate_scores = model(candidate_tensor.to(device)).cpu().numpy()

    candidate_feature_matrix = candidate_tensor.numpy()

    anchor_vectors: List[np.ndarray] = []
    if anchors:
        anchor_tensor = featurizer.transform(sorted(anchors)).numpy()
        if anchor_tensor.size:
            unique_anchor_vectors = np.unique(anchor_tensor, axis=0)
            anchor_vectors = [vec for vec in unique_anchor_vectors]

    selected_indices, selected_scores, selected_diversity = pareto_selection(
        candidate_strings,
        candidate_scores,
        candidate_feature_matrix,
        anchor_vectors,
        args.m,
    )

    selected_rows = []
    for rank, (idx, score, diversity_value) in enumerate(
        zip(selected_indices, selected_scores, selected_diversity), start=1
    ):
        selected_rows.append(
            {
                "rank": rank,
                "mutations": candidate_strings[idx],
                "predicted_score": score,
                "diversity": diversity_value,
            }
        )

    selected_df = pd.DataFrame(selected_rows)
    args.out_selected.parent.mkdir(parents=True, exist_ok=True)
    selected_df.to_csv(args.out_selected, index=False, sep="\t")
    if args.verbose:
        print(f"Wrote {len(selected_df)} selections to {args.out_selected}")


if __name__ == "__main__":
    main()

