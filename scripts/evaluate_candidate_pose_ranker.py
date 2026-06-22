from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset.resolve()
    model_dir = args.model_dir.resolve()
    output_dir = unique_dir(args.output.resolve())
    output_dir.mkdir(parents=True, exist_ok=False)

    rows = read_csv(dataset_dir / "candidate_pose_examples.csv")
    if not rows:
        raise SystemExit(f"No candidate_pose_examples.csv rows found under {dataset_dir}")

    model = load_npz(model_dir / "candidate_pose_rank_net.npz")
    schema = json.loads((model_dir / "candidate_pose_rank_net_schema.json").read_text(encoding="utf-8"))
    scored = score_rows(rows, model, schema)
    eval_rows = evaluate_groups(scored)
    summary = summarize(eval_rows, rows)

    write_csv(output_dir / "candidate_pose_scores.csv", scored)
    write_csv(output_dir / "group_eval.csv", eval_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    write_readme(output_dir, summary)
    print(output_dir)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CandidatePoseRankNet on candidate-pose examples.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = path.with_name(f"{path.name}_{stamp}")
    suffix = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.name}_{stamp}_{suffix:02d}")
        suffix += 1
    return candidate


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in data.files}


def score_rows(rows: list[dict[str, str]], model: dict[str, np.ndarray], schema: dict[str, Any]) -> list[dict[str, Any]]:
    x = build_matrix(rows, schema)
    hidden = np.maximum(x @ model["w1"] + model["b1"], 0.0)
    logits = hidden @ model["w2"] + model["b2"]
    probs = sigmoid(logits).reshape(-1)
    scored: list[dict[str, Any]] = []
    for row, prob in zip(rows, probs):
        item = dict(row)
        item["ranker_prob"] = float(prob)
        scored.append(item)
    return scored


def build_matrix(rows: list[dict[str, str]], schema: dict[str, Any]) -> np.ndarray:
    x_rows: list[list[float]] = []
    for row in rows:
        features: list[float] = []
        for column in schema["numeric_columns"]:
            raw = row.get(column, "")
            features.append(parse_float(raw))
            features.append(0.0 if is_float(raw) else 1.0)
        for column in schema["categorical_columns"]:
            value = str(row.get(column, ""))
            for category in schema["categories"][column]:
                features.append(1.0 if value == category else 0.0)
        x_rows.append(features)
    x = np.asarray(x_rows, dtype=np.float64)
    mean = np.asarray(schema["feature_mean"], dtype=np.float64).reshape(1, -1)
    std = np.asarray(schema["feature_std"], dtype=np.float64).reshape(1, -1)
    return (x - mean) / std


def evaluate_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("run_name", "")),
            str(row.get("target_name", "")),
            str(row.get("strategy", "")),
            str(row.get("gravity", "")),
            str(row.get("trial", "")),
            str(row.get("slot_id", "")),
            str(row.get("candidate_rock_index", "")),
        )
        groups[key].append(row)

    output: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        selected_ids = {str(row.get("candidate_id", "")) for row in items if parse_float(row.get("label_selected_by_pose_search", "0")) > 0.5}
        ranked_by_net = sorted(items, key=lambda row: -float(row["ranker_prob"]))
        ranked_by_score = sorted(items, key=lambda row: parse_float(row.get("candidate_score", "999999")))
        top1_net = {str(row.get("candidate_id", "")) for row in ranked_by_net[:1]}
        top3_net = {str(row.get("candidate_id", "")) for row in ranked_by_net[:3]}
        top1_score = {str(row.get("candidate_id", "")) for row in ranked_by_score[:1]}
        output.append(
            {
                "run_name": key[0],
                "target_name": key[1],
                "strategy": key[2],
                "gravity": key[3],
                "trial": key[4],
                "slot_id": key[5],
                "candidate_rock_index": key[6],
                "candidate_count": len(items),
                "selected_candidate_ids": " ".join(sorted(selected_ids)),
                "net_top1_hit": int(bool(selected_ids.intersection(top1_net))),
                "net_top3_hit": int(bool(selected_ids.intersection(top3_net))),
                "hand_score_top1_hit": int(bool(selected_ids.intersection(top1_score))),
                "net_top1_candidate_id": ranked_by_net[0].get("candidate_id", "") if ranked_by_net else "",
                "net_top1_prob": float(ranked_by_net[0]["ranker_prob"]) if ranked_by_net else 0.0,
            }
        )
    return output


def summarize(eval_rows: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    group_count = max(len(eval_rows), 1)
    by_gravity: dict[str, dict[str, Any]] = {}
    for gravity in sorted({str(row["gravity"]) for row in eval_rows}):
        subset = [row for row in eval_rows if row["gravity"] == gravity]
        count = max(len(subset), 1)
        by_gravity[gravity] = {
            "groups": len(subset),
            "net_top1_hit_rate": sum(int(row["net_top1_hit"]) for row in subset) / count,
            "net_top3_hit_rate": sum(int(row["net_top3_hit"]) for row in subset) / count,
            "hand_score_top1_hit_rate": sum(int(row["hand_score_top1_hit"]) for row in subset) / count,
        }
    return {
        "candidate_pose_rows": len(rows),
        "groups": len(eval_rows),
        "net_top1_hit_rate": sum(int(row["net_top1_hit"]) for row in eval_rows) / group_count,
        "net_top3_hit_rate": sum(int(row["net_top3_hit"]) for row in eval_rows) / group_count,
        "hand_score_top1_hit_rate": sum(int(row["hand_score_top1_hit"]) for row in eval_rows) / group_count,
        "by_gravity": by_gravity,
        "note": "Labels are current hand-coded pose-search selections, not independent physical truth.",
    }


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def parse_float(value: str | None) -> float:
    return float(value) if is_float(value) else 0.0


def is_float(value: str | None) -> bool:
    if value in {"", None}:
        return False
    try:
        float(value)
    except ValueError:
        return False
    return True


def write_readme(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# CandidatePoseRankNet Evaluation",
        "",
        "This evaluates whether the learned ranker can recover the current hand-coded pose-search choice within each slot candidate group.",
        "",
        f"- candidate pose rows: {summary['candidate_pose_rows']}",
        f"- groups: {summary['groups']}",
        f"- net top-1 hit rate: {summary['net_top1_hit_rate']:.3f}",
        f"- net top-3 hit rate: {summary['net_top3_hit_rate']:.3f}",
        f"- hand-score top-1 hit rate: {summary['hand_score_top1_hit_rate']:.3f}",
        "",
        "The hand-score baseline uses post-settle metrics and is not available before simulation. The learned ranker uses only pre-pose tabular features from its schema.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
