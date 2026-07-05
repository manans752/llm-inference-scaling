from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def summarize_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "num_examples": 0,
            "accuracy": 0.0,
            "invalid_output_rate": 0.0,
            "parse_error_rate": 0.0,
            "avg_total_tokens": 0.0,
            "total_tokens": 0,
            "avg_latency_seconds": 0.0,
        }

    frame = pd.DataFrame(records)
    summary = {
        "num_examples": int(len(frame)),
        "accuracy": float(frame["correct"].mean()),
        "invalid_output_rate": float((frame["error_type"] == "invalid_output").mean()),
        "parse_error_rate": float((frame["error_type"] == "parse_error").mean()),
        "avg_total_tokens": float(frame["total_tokens"].mean()),
        "total_tokens": int(frame["total_tokens"].sum()),
        "avg_latency_seconds": float(frame["latency_seconds"].mean()),
    }

    if "category" in frame.columns:
        by_category = frame.groupby("category")["correct"].mean().sort_values(ascending=False)
        summary["accuracy_by_category"] = {str(key): float(value) for key, value in by_category.items()}

    if summary["total_tokens"] > 0:
        summary["accuracy_per_1k_tokens"] = summary["accuracy"] / (summary["total_tokens"] / 1000.0)
    else:
        summary["accuracy_per_1k_tokens"] = 0.0

    return summary


def save_experiment_results(
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
    output_dir: str | Path,
    run_name: str,
) -> dict[str, Path]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_run_name = _safe_name(run_name)

    rows_path = target_dir / f"{safe_run_name}.json"
    csv_path = target_dir / f"{safe_run_name}.csv"
    summary_path = target_dir / f"{safe_run_name}_summary.json"

    rows_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    pd.DataFrame(records).to_csv(csv_path, index=False)

    summary_payload = {
        "metadata": metadata,
        "summary": summarize_results(records),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return {
        "json": rows_path,
        "csv": csv_path,
        "summary": summary_path,
    }
