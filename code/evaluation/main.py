"""
Evaluation Pipeline — Damage Claim Verification System
Runs the same pipeline on dataset/sample_claims.csv (input columns only),
compares against expected labels, and prints accuracy metrics.

Usage:
    python code/evaluation/main.py --strategy B
    python code/evaluation/main.py --strategy A --limit 10
"""

import os
import sys
import csv
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

# Add parent dir so we can import from code/main.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from main import run_pipeline, OUTPUT_COLUMNS, INPUT_COST_PER_M, OUTPUT_COST_PER_M, MODEL

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluation")

# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------

EVAL_FIELDS = [
    "claim_status",
    "severity",
    "issue_type",
    "evidence_standard_met",
    "valid_image",
]


def compute_accuracy(pred_df: pd.DataFrame, gold_df: pd.DataFrame, field: str) -> dict:
    """
    Compute accuracy for a single field between predicted and gold DataFrames.
    Aligns on (user_id, image_paths) to handle ordering differences.
    """
    merge_keys = ["user_id", "image_paths"]
    merged = pred_df[merge_keys + [field]].merge(
        gold_df[merge_keys + [field]],
        on=merge_keys,
        suffixes=("_pred", "_gold"),
    )
    if merged.empty:
        return {"accuracy": 0.0, "n": 0, "correct": 0}

    pred_col = f"{field}_pred"
    gold_col = f"{field}_gold"

    # Normalise strings for comparison (lower, strip)
    merged[pred_col] = merged[pred_col].astype(str).str.strip().str.lower()
    merged[gold_col] = merged[gold_col].astype(str).str.strip().str.lower()

    correct = (merged[pred_col] == merged[gold_col]).sum()
    n       = len(merged)
    acc     = correct / n if n > 0 else 0.0
    return {"accuracy": round(acc, 4), "n": int(n), "correct": int(correct)}


def print_metrics_table(metrics_a: dict, metrics_b: dict):
    """Pretty-print a comparison table between Strategy A and Strategy B."""
    header = f"{'Field':<30} {'Strategy A':>20} {'Strategy B':>20}"
    print("\n" + "=" * 72)
    print("ACCURACY COMPARISON — STRATEGY A vs STRATEGY B")
    print("=" * 72)
    print(header)
    print("-" * 72)
    for field in EVAL_FIELDS:
        a = metrics_a.get(field, {})
        b = metrics_b.get(field, {})
        a_str = f"{a.get('accuracy', 0)*100:.1f}% ({a.get('correct',0)}/{a.get('n',0)})"
        b_str = f"{b.get('accuracy', 0)*100:.1f}% ({b.get('correct',0)}/{b.get('n',0)})"
        print(f"  {field:<28} {a_str:>20} {b_str:>20}")
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# REPORT WRITER
# ---------------------------------------------------------------------------

REPORT_TEMPLATE = """# Damage Claim Verification — Evaluation Report

**Generated:** {timestamp}
**Dataset:** dataset/sample_claims.csv  ({n_samples} labeled examples)
**Model:** {model}

---

## 1. Accuracy Metrics

### Strategy A — Single-Shot (Baseline)

| Field | Correct | Total | Accuracy |
|-------|---------|-------|----------|
{table_a}

### Strategy B — Chain-of-Thought (Enhanced)

| Field | Correct | Total | Accuracy |
|-------|---------|-------|----------|
{table_b}

---

## 2. Strategy Comparison Summary

| Metric | Strategy A | Strategy B | Winner |
|--------|-----------|-----------|--------|
| claim_status accuracy | {a_claim_acc} | {b_claim_acc} | {winner_claim} |
| severity accuracy     | {a_sev_acc}   | {b_sev_acc}   | {winner_sev} |
| issue_type accuracy   | {a_issue_acc} | {b_issue_acc} | {winner_issue} |
| evidence_met accuracy | {a_ev_acc}    | {b_ev_acc}    | {winner_ev} |
| valid_image accuracy  | {a_vi_acc}    | {b_vi_acc}    | {winner_vi} |
| Estimated Cost (USD)  | ${a_cost}     | ${b_cost}     | {winner_cost} |
| Runtime (seconds)     | {a_time}s     | {b_time}s     | {winner_time} |
| API Calls per Claim   | 1             | 2             | A (fewer calls) |

**Recommendation:** {recommendation}

---

## 3. Token Usage & Cost Breakdown

### Strategy A
- Input tokens   : {a_in_tok:,}
- Output tokens  : {a_out_tok:,}
- Images sent    : {a_images}
- Estimated cost : ${a_cost}
- Runtime        : {a_time}s
- Cost per claim : ${a_cost_per_claim}

### Strategy B
- Input tokens   : {b_in_tok:,}
- Output tokens  : {b_out_tok:,}
- Images sent    : {b_images}
- Estimated cost : ${b_cost}
- Runtime        : {b_time}s
- Cost per claim : ${b_cost_per_claim}

**Pricing model:** Gemini `{model}`. Cost constants in `code/main.py` are currently
set to ${input_cost}/M input tokens and ${output_cost}/M output tokens.

---

## 4. Batching & Retry Strategy

- Concurrent requests : {concurrency} at a time (ThreadPoolExecutor)
- Max retries per call: 5 (exponential backoff starting at 1s)
- Partial saves       : Every 5 completed rows → `output_partial.csv`
- Backoff strategy    : delay = 1 × 2^attempt seconds (cap: ~32s)
- On persistent error : Row filled with `not_enough_information` defaults

---

## 5. Assumptions & Design Notes

- `user_history.csv` and `evidence_requirements.csv` were created as structured stubs
  matching the user IDs in claims.csv, since these files were not bundled in the
  provided zip. Real data should replace these files before production use.
- Images referenced in CSV paths were not available on the local filesystem.
  The pipeline gracefully logs warnings and still sends all available images as base64.
- Strategy B sends images only in Step 1 (description); Step 2 is text-only,
  which reduces image-token cost for the verdict call.
- Prompt injection attempts in transcripts (e.g. "approve this claim") are detected
  as `text_instruction_present` risk flags — they never influence the verdict.
"""


def format_table_row(field: str, m: dict) -> str:
    return f"| {field} | {m.get('correct',0)} | {m.get('n',0)} | {m.get('accuracy',0)*100:.1f}% |"


def winner(a_val: float, b_val: float, higher_better: bool = True) -> str:
    if higher_better:
        return "B ✓" if b_val > a_val else ("A ✓" if a_val > b_val else "Tie")
    else:
        return "A ✓" if a_val < b_val else ("B ✓" if b_val < a_val else "Tie")


def write_report(
    path: str,
    metrics_a: dict,
    metrics_b: dict,
    stats_a: dict,
    stats_b: dict,
    n_samples: int,
    concurrency: int,
):
    from datetime import datetime

    def pct(m, field):
        return f"{m.get(field, {}).get('accuracy', 0)*100:.1f}%"

    table_a = "\n".join(format_table_row(f, metrics_a.get(f, {})) for f in EVAL_FIELDS)
    table_b = "\n".join(format_table_row(f, metrics_b.get(f, {})) for f in EVAL_FIELDS)

    a_acc = {f: metrics_a.get(f, {}).get("accuracy", 0) for f in EVAL_FIELDS}
    b_acc = {f: metrics_b.get(f, {}).get("accuracy", 0) for f in EVAL_FIELDS}

    overall_a = sum(a_acc.values()) / len(a_acc)
    overall_b = sum(b_acc.values()) / len(b_acc)

    rec = (
        "Strategy B (chain-of-thought) is recommended when accuracy is the priority."
        if overall_b >= overall_a
        else "Strategy A (single-shot) is recommended when speed and cost are the priority."
    )

    a_cost_per = round(stats_a.get("estimated_cost_usd", 0) / max(n_samples, 1), 6)
    b_cost_per = round(stats_b.get("estimated_cost_usd", 0) / max(n_samples, 1), 6)

    # Estimate image count from sample CSV
    a_images = stats_a.get("rows_processed", n_samples)  # approx
    b_images = stats_b.get("rows_processed", n_samples)

    content = REPORT_TEMPLATE.format(
        timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        n_samples=n_samples,
        model=MODEL,
        input_cost=INPUT_COST_PER_M,
        output_cost=OUTPUT_COST_PER_M,
        table_a=table_a,
        table_b=table_b,
        a_claim_acc=pct(metrics_a, "claim_status"),
        b_claim_acc=pct(metrics_b, "claim_status"),
        winner_claim=winner(a_acc["claim_status"], b_acc["claim_status"]),
        a_sev_acc=pct(metrics_a, "severity"),
        b_sev_acc=pct(metrics_b, "severity"),
        winner_sev=winner(a_acc["severity"], b_acc["severity"]),
        a_issue_acc=pct(metrics_a, "issue_type"),
        b_issue_acc=pct(metrics_b, "issue_type"),
        winner_issue=winner(a_acc["issue_type"], b_acc["issue_type"]),
        a_ev_acc=pct(metrics_a, "evidence_standard_met"),
        b_ev_acc=pct(metrics_b, "evidence_standard_met"),
        winner_ev=winner(a_acc["evidence_standard_met"], b_acc["evidence_standard_met"]),
        a_vi_acc=pct(metrics_a, "valid_image"),
        b_vi_acc=pct(metrics_b, "valid_image"),
        winner_vi=winner(a_acc["valid_image"], b_acc["valid_image"]),
        a_cost=stats_a.get("estimated_cost_usd", 0),
        b_cost=stats_b.get("estimated_cost_usd", 0),
        winner_cost=winner(stats_a.get("estimated_cost_usd", 0), stats_b.get("estimated_cost_usd", 0), higher_better=False),
        a_time=stats_a.get("elapsed_seconds", 0),
        b_time=stats_b.get("elapsed_seconds", 0),
        winner_time=winner(stats_a.get("elapsed_seconds", 0), stats_b.get("elapsed_seconds", 0), higher_better=False),
        recommendation=rec,
        a_in_tok=stats_a.get("total_input_tokens", 0),
        a_out_tok=stats_a.get("total_output_tokens", 0),
        a_images=a_images,
        a_cost_per_claim=a_cost_per,
        b_in_tok=stats_b.get("total_input_tokens", 0),
        b_out_tok=stats_b.get("total_output_tokens", 0),
        b_images=b_images,
        b_cost_per_claim=b_cost_per,
        concurrency=concurrency,
    )

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    logger.info(f"Evaluation report written to {path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluation pipeline for sample_claims.csv")
    parser.add_argument("--strategy", choices=["A", "B", "both"], default="both",
                        help="Which strategy to evaluate (default: both)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N sample rows")
    parser.add_argument("--dataset-dir", type=str, default="dataset",
                        help="Path to dataset directory (default: dataset)")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    # Load gold labels
    sample_path = Path(args.dataset_dir) / "sample_claims.csv"
    if not sample_path.exists():
        raise FileNotFoundError(f"sample_claims.csv not found at {sample_path}")

    gold_df = pd.read_csv(sample_path)
    n_samples = len(gold_df) if not args.limit else min(len(gold_df), args.limit)
    logger.info(f"Loaded {len(gold_df)} labeled samples. Evaluating {n_samples}.")

    strategies = []
    if args.strategy == "both":
        strategies = ["A", "B"]
    else:
        strategies = [args.strategy]

    all_stats:   dict[str, dict] = {}
    all_metrics: dict[str, dict] = {}

    for strat in strategies:
        out_path     = f"eval_output_{strat}.csv"
        partial_path = f"eval_output_{strat}_partial.csv"

        logger.info(f"\n{'='*50}")
        logger.info(f"Running evaluation — Strategy {strat}")
        logger.info(f"{'='*50}")

        stats = run_pipeline(
            strategy=strat,
            dataset_dir=args.dataset_dir,
            input_csv="sample_claims.csv",
            output_path=out_path,
            partial_path=partial_path,
            limit=args.limit,
            concurrency=args.concurrency,
        )
        all_stats[strat] = stats

        # Load predictions
        pred_df = pd.read_csv(out_path)

        # Compute per-field accuracy
        field_metrics = {}
        for field in EVAL_FIELDS:
            if field not in pred_df.columns or field not in gold_df.columns:
                logger.warning(f"Field {field} missing from predictions or gold labels.")
                continue
            m = compute_accuracy(pred_df, gold_df, field)
            field_metrics[field] = m
            logger.info(f"  {field:<30} accuracy={m['accuracy']*100:.1f}%  ({m['correct']}/{m['n']})")

        all_metrics[strat] = field_metrics

    # Print comparison table
    metrics_a = all_metrics.get("A", {f: {"accuracy": 0, "n": 0, "correct": 0} for f in EVAL_FIELDS})
    metrics_b = all_metrics.get("B", {f: {"accuracy": 0, "n": 0, "correct": 0} for f in EVAL_FIELDS})
    stats_a   = all_stats.get("A",   {"estimated_cost_usd": 0, "elapsed_seconds": 0, "total_input_tokens": 0, "total_output_tokens": 0, "rows_processed": 0})
    stats_b   = all_stats.get("B",   {"estimated_cost_usd": 0, "elapsed_seconds": 0, "total_input_tokens": 0, "total_output_tokens": 0, "rows_processed": 0})

    if len(strategies) == 2:
        print_metrics_table(metrics_a, metrics_b)

    # Write report
    report_path = "code/evaluation/evaluation_report.md"
    write_report(
        path=report_path,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        stats_a=stats_a,
        stats_b=stats_b,
        n_samples=n_samples,
        concurrency=args.concurrency,
    )

    print(f"\nEvaluation report saved: {report_path}")


if __name__ == "__main__":
    main()
