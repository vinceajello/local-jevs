#!/usr/bin/env python3
"""Benchmark and Evaluation Script for Laya System 1 Decision Models.

Evaluates trained Laya models or base Laya models on benchmark datasets
(local JSON/JSONL like benchmark.json or Hugging Face datasets like LocalLLaMA/typed-decisions).

Supports:
- Evaluation of custom checkpoints (e.g. ./my_laya_model)
- Side-by-side comparison with the base Laya model (--compare_base)
- Multi-dimensional metrics: Accuracy, MAE, Brier Score, Log Loss, and Latency (mean/p95)
- Breakdown by question type (choice, score, noul)
- Console table reporting and JSON export
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from laya import Agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark and evaluate Laya decision models against baseline benchmarks."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./my_laya_model" if os.path.exists("./my_laya_model") else "convaiinnovations/laya",
        help="Path to trained model directory or Hugging Face model id (default: ./my_laya_model if exists, else convaiinnovations/laya).",
    )
    parser.add_argument(
        "--compare_base",
        action="store_true",
        help="Also benchmark the base Laya model (convaiinnovations/laya) side-by-side for comparison.",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="convaiinnovations/laya",
        help="Hugging Face repo or path for the base model when using --compare_base (default: convaiinnovations/laya).",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to local JSON or JSONL benchmark dataset (e.g. src/laya/benchmark.json). If omitted, defaults to --dataset_name.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="LocalLLaMA/typed-decisions",
        help="Hugging Face benchmark dataset name (default: 'LocalLLaMA/typed-decisions'). Used when --dataset_path is not specified.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="all",
        help="Dataset configuration name for Hugging Face datasets (default: 'all').",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="test",
        help="Dataset split to evaluate for Hugging Face datasets (default: 'test').",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of benchmark cases to evaluate (default: evaluate all).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('mps', 'cuda', 'cpu', or auto-detect).",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save full evaluation results and metrics as a JSON file.",
    )
    return parser.parse_args()


def get_default_device(requested_device: Optional[str]) -> str:
    if requested_device:
        return requested_device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_benchmark_cases(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], str]:
    """Load evaluation cases from local file or Hugging Face dataset."""
    cases: List[Dict[str, Any]] = []
    source_desc = ""

    if args.dataset_path:
        path = args.dataset_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Local benchmark dataset not found: {path}")
        source_desc = f"local file '{path}'"
        with open(path, "r", encoding="utf-8") as f:
            if path.endswith(".jsonl"):
                for line in f:
                    line = line.strip()
                    if line:
                        cases.append(json.loads(line))
            else:
                data = json.load(f)
                cases = data if isinstance(data, list) else [data]

    else:
        # Default: Hugging Face dataset (e.g. LocalLLaMA/typed-decisions)
        dataset_name = args.dataset_name or "LocalLLaMA/typed-decisions"
        source_desc = f"Hugging Face dataset '{dataset_name}' (split: {args.dataset_split})"
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Package 'datasets' is required for Hugging Face benchmark datasets. "
                "Install it using: uv add datasets or pip install datasets"
            )
        try:
            if args.dataset_config:
                ds = load_dataset(dataset_name, args.dataset_config, split=args.dataset_split)
            else:
                ds = load_dataset(dataset_name, split=args.dataset_split)
        except Exception:
            try:
                # Try fallback config 'all'
                ds = load_dataset(dataset_name, "all", split=args.dataset_split)
            except Exception:
                # Fallback without config
                ds = load_dataset(dataset_name, split=args.dataset_split)

        for row in ds:
            state = json.loads(row["state"]) if isinstance(row.get("state"), str) else row.get("state")
            questions = (
                json.loads(row["questions"]) if isinstance(row.get("questions"), str) else row.get("questions")
            )
            gold = json.loads(row["gold"]) if isinstance(row.get("gold"), str) else row.get("gold")
            cases.append({"id": row.get("id"), "state": state, "questions": questions, "gold": gold})

    if args.limit and len(cases) > args.limit:
        cases = cases[: args.limit]

    return cases, source_desc


def evaluate_model(
    agent: Agent,
    cases: List[Dict[str, Any]],
    model_name: str,
) -> Dict[str, Any]:
    """Run inference over benchmark cases and compute metrics."""
    latencies: List[float] = []
    
    # Aggregated metrics counters
    choice_results = {"correct": 0, "total": 0, "brier_sum": 0.0, "logloss_sum": 0.0}
    score_results = {"exact_match": 0, "total": 0, "mae_sum": 0.0, "brier_sum": 0.0}
    noul_results = {"correct": 0, "total": 0, "brier_sum": 0.0, "logloss_sum": 0.0}

    detailed_cases = []

    for idx, case in enumerate(cases):
        state = case.get("state", {})
        questions = case.get("questions", {})
        gold = case.get("gold", case.get("answers", {}))

        t0 = time.perf_counter()
        pred = agent.predict(state, questions)
        t_elapsed = (time.perf_counter() - t0) * 1000.0  # ms
        latencies.append(t_elapsed)

        answers = pred.get("answers", {})
        case_report = {
            "id": case.get("id", f"case_{idx}"),
            "latency_ms": round(t_elapsed, 2),
            "evaluations": {},
        }

        for qid, q_def in questions.items():
            if qid not in gold or qid not in answers:
                continue
            q_type = q_def.get("type")
            q_pred = answers[qid]
            q_gold = gold[qid]

            if q_type == "choice":
                target_choice = q_gold.get("choice")
                pred_choice = q_pred.get("choice")
                pred_probs = q_pred.get("probabilities", {})

                is_correct = 1 if pred_choice == target_choice else 0
                choice_results["total"] += 1
                choice_results["correct"] += is_correct

                # Brier score: sum_k (p_k - y_k)^2
                # Log loss: -log(p_target)
                brier = 0.0
                all_keys = set(pred_probs.keys())
                if target_choice:
                    all_keys.add(target_choice)

                for k in all_keys:
                    p = pred_probs.get(k, 0.0)
                    y = 1.0 if k == target_choice else 0.0
                    brier += (p - y) ** 2

                p_target = pred_probs.get(target_choice, 1e-12)
                logloss = -math.log(max(p_target, 1e-12))

                choice_results["brier_sum"] += brier
                choice_results["logloss_sum"] += logloss

                case_report["evaluations"][qid] = {
                    "type": "choice",
                    "pred": pred_choice,
                    "target": target_choice,
                    "correct": bool(is_correct),
                    "confidence": q_pred.get("confidence"),
                    "brier": round(brier, 4),
                    "logloss": round(logloss, 4),
                }

            elif q_type == "score":
                target_score = float(q_gold.get("score", 0.0))
                pred_score = float(q_pred.get("score", 0.0))
                pred_probs = q_pred.get("probabilities", {})

                is_exact = 1 if round(pred_score) == round(target_score) else 0
                mae = abs(pred_score - target_score)

                score_results["total"] += 1
                score_results["exact_match"] += is_exact
                score_results["mae_sum"] += mae

                # Brier score across score classes
                target_key = str(int(round(target_score)))
                brier = 0.0
                all_keys = set(pred_probs.keys())
                all_keys.add(target_key)
                for k in all_keys:
                    p = pred_probs.get(k, 0.0)
                    y = 1.0 if k == target_key else 0.0
                    brier += (p - y) ** 2
                score_results["brier_sum"] += brier

                case_report["evaluations"][qid] = {
                    "type": "score",
                    "pred_score": round(pred_score, 3),
                    "target_score": target_score,
                    "exact_match": bool(is_exact),
                    "mae": round(mae, 3),
                    "brier": round(brier, 4),
                }

            elif q_type == "noul":
                # noul is probability of positive decision
                if isinstance(q_gold, dict):
                    target_val = float(q_gold.get("noul", 0.0))
                else:
                    target_val = float(q_gold)

                pred_val = float(q_pred.get("noul", 0.0))
                target_binary = 1.0 if target_val >= 0.5 else 0.0
                pred_binary = 1.0 if pred_val >= 0.5 else 0.0

                is_correct = 1 if pred_binary == target_binary else 0
                brier = (pred_val - target_binary) ** 2
                eps = 1e-12
                p_clamped = min(max(pred_val, eps), 1.0 - eps)
                logloss = -(target_binary * math.log(p_clamped) + (1.0 - target_binary) * math.log(1.0 - p_clamped))

                noul_results["total"] += 1
                noul_results["correct"] += is_correct
                noul_results["brier_sum"] += brier
                noul_results["logloss_sum"] += logloss

                case_report["evaluations"][qid] = {
                    "type": "noul",
                    "pred_prob": round(pred_val, 4),
                    "target_prob": round(target_val, 4),
                    "correct": bool(is_correct),
                    "brier": round(brier, 4),
                    "logloss": round(logloss, 4),
                }

        detailed_cases.append(case_report)

    # Compute summaries
    total_decisions = choice_results["total"] + score_results["total"] + noul_results["total"]
    total_correct = choice_results["correct"] + score_results["exact_match"] + noul_results["correct"]
    overall_acc = (total_correct / total_decisions * 100.0) if total_decisions > 0 else 0.0

    choice_acc = (choice_results["correct"] / choice_results["total"] * 100.0) if choice_results["total"] > 0 else 0.0
    choice_brier = (choice_results["brier_sum"] / choice_results["total"]) if choice_results["total"] > 0 else 0.0
    choice_logloss = (choice_results["logloss_sum"] / choice_results["total"]) if choice_results["total"] > 0 else 0.0

    score_match = (score_results["exact_match"] / score_results["total"] * 100.0) if score_results["total"] > 0 else 0.0
    score_mae = (score_results["mae_sum"] / score_results["total"]) if score_results["total"] > 0 else 0.0
    score_brier = (score_results["brier_sum"] / score_results["total"]) if score_results["total"] > 0 else 0.0

    noul_acc = (noul_results["correct"] / noul_results["total"] * 100.0) if noul_results["total"] > 0 else 0.0
    noul_brier = (noul_results["brier_sum"] / noul_results["total"]) if noul_results["total"] > 0 else 0.0
    noul_logloss = (noul_results["logloss_sum"] / noul_results["total"]) if noul_results["total"] > 0 else 0.0

    mean_brier = (
        (choice_results["brier_sum"] + score_results["brier_sum"] + noul_results["brier_sum"]) / total_decisions
        if total_decisions > 0
        else 0.0
    )

    valid_logloss_total = choice_results["total"] + noul_results["total"]
    mean_logloss = (
        (choice_results["logloss_sum"] + noul_results["logloss_sum"]) / valid_logloss_total
        if valid_logloss_total > 0
        else 0.0
    )

    lat_np = np.array(latencies) if latencies else np.array([0.0])
    mean_lat = float(np.mean(lat_np))
    p50_lat = float(np.median(lat_np))
    p95_lat = float(np.percentile(lat_np, 95))
    throughput = (1000.0 / mean_lat) if mean_lat > 0 else 0.0

    return {
        "model_name": model_name,
        "cases_evaluated": len(cases),
        "decisions_evaluated": total_decisions,
        "overall_accuracy_pct": round(overall_acc, 2),
        "mean_brier_score": round(mean_brier, 4),
        "mean_log_loss": round(mean_logloss, 4),
        "choice": {
            "total": choice_results["total"],
            "accuracy_pct": round(choice_acc, 2),
            "mean_brier": round(choice_brier, 4),
            "mean_logloss": round(choice_logloss, 4),
        },
        "score": {
            "total": score_results["total"],
            "exact_match_pct": round(score_match, 2),
            "mae": round(score_mae, 4),
            "mean_brier": round(score_brier, 4),
        },
        "noul": {
            "total": noul_results["total"],
            "accuracy_pct": round(noul_acc, 2),
            "mean_brier": round(noul_brier, 4),
            "mean_logloss": round(noul_logloss, 4),
        },
        "latency": {
            "mean_ms": round(mean_lat, 2),
            "p50_ms": round(p50_lat, 2),
            "p95_ms": round(p95_lat, 2),
            "throughput_qps": round(throughput, 1),
        },
        "details": detailed_cases,
    }


def print_single_summary(m: Dict[str, Any]):
    print("\n" + "=" * 72)
    print(f"  LAYA BENCHMARK EVALUATION RESULTS: {m['model_name']}")
    print("=" * 72)
    print(f"  Benchmark Cases Evaluated  : {m['cases_evaluated']}")
    print(f"  Total Decisions Evaluated  : {m['decisions_evaluated']}")
    print("-" * 72)
    print(f"  Overall Decision Accuracy  : {m['overall_accuracy_pct']}%")
    print(f"  Mean Brier Score           : {m['mean_brier_score']}  (lower is better)")
    print(f"  Mean Log Loss (NLL)        : {m['mean_log_loss']}  (lower is better)")
    print("-" * 72)
    print(f"  Choice Classification      : {m['choice']['accuracy_pct']}% Acc | Brier: {m['choice']['mean_brier']} | NLL: {m['choice']['mean_logloss']} (n={m['choice']['total']})")
    print(f"  Score Estimation           : {m['score']['exact_match_pct']}% Match | MAE: {m['score']['mae']} | Brier: {m['score']['mean_brier']} (n={m['score']['total']})")
    print(f"  Noul (Binary) Decisions    : {m['noul']['accuracy_pct']}% Acc | Brier: {m['noul']['mean_brier']} | NLL: {m['noul']['mean_logloss']} (n={m['noul']['total']})")
    print("-" * 72)
    print(f"  Inference Latency (Mean)   : {m['latency']['mean_ms']} ms")
    print(f"  Inference Latency (p95)    : {m['latency']['p95_ms']} ms")
    print(f"  Throughput                 : {m['latency']['throughput_qps']} queries/sec")
    print("=" * 72 + "\n")


def print_comparison_table(base_m: Dict[str, Any], fine_m: Dict[str, Any]):
    print("\n" + "=" * 80)
    print("       LAYA BENCHMARK COMPARISON: BASE MODEL vs. FINE-TUNED MODEL")
    print("=" * 80)
    header = f"{'Metric':<30} | {'Base Model':<16} | {'Fine-Tuned Model':<16} | {'Delta':<12}"
    print(header)
    print("-" * 80)

    rows = [
        (
            "Overall Accuracy (%)",
            f"{base_m['overall_accuracy_pct']:.2f}%",
            f"{fine_m['overall_accuracy_pct']:.2f}%",
            f"{fine_m['overall_accuracy_pct'] - base_m['overall_accuracy_pct']:+.2f}%",
        ),
        (
            "Mean Brier Score (lower=better)",
            f"{base_m['mean_brier_score']:.4f}",
            f"{fine_m['mean_brier_score']:.4f}",
            f"{fine_m['mean_brier_score'] - base_m['mean_brier_score']:+.4f}",
        ),
        (
            "Mean Log Loss (lower=better)",
            f"{base_m['mean_log_loss']:.4f}",
            f"{fine_m['mean_log_loss']:.4f}",
            f"{fine_m['mean_log_loss'] - base_m['mean_log_loss']:+.4f}",
        ),
        (
            "Choice Accuracy (%)",
            f"{base_m['choice']['accuracy_pct']:.2f}%",
            f"{fine_m['choice']['accuracy_pct']:.2f}%",
            f"{fine_m['choice']['accuracy_pct'] - base_m['choice']['accuracy_pct']:+.2f}%",
        ),
        (
            "Score MAE (lower=better)",
            f"{base_m['score']['mae']:.4f}",
            f"{fine_m['score']['mae']:.4f}",
            f"{fine_m['score']['mae'] - base_m['score']['mae']:+.4f}",
        ),
        (
            "Score Exact Match (%)",
            f"{base_m['score']['exact_match_pct']:.2f}%",
            f"{fine_m['score']['exact_match_pct']:.2f}%",
            f"{fine_m['score']['exact_match_pct'] - base_m['score']['exact_match_pct']:+.2f}%",
        ),
        (
            "Noul Binary Accuracy (%)",
            f"{base_m['noul']['accuracy_pct']:.2f}%",
            f"{fine_m['noul']['accuracy_pct']:.2f}%",
            f"{fine_m['noul']['accuracy_pct'] - base_m['noul']['accuracy_pct']:+.2f}%",
        ),
        (
            "Inference Latency Mean (ms)",
            f"{base_m['latency']['mean_ms']:.2f} ms",
            f"{fine_m['latency']['mean_ms']:.2f} ms",
            f"{fine_m['latency']['mean_ms'] - base_m['latency']['mean_ms']:+.2f} ms",
        ),
        (
            "Inference Latency p95 (ms)",
            f"{base_m['latency']['p95_ms']:.2f} ms",
            f"{fine_m['latency']['p95_ms']:.2f} ms",
            f"{fine_m['latency']['p95_ms'] - base_m['latency']['p95_ms']:+.2f} ms",
        ),
        (
            "Throughput (queries/sec)",
            f"{base_m['latency']['throughput_qps']:.1f} qps",
            f"{fine_m['latency']['throughput_qps']:.1f} qps",
            f"{fine_m['latency']['throughput_qps'] - base_m['latency']['throughput_qps']:+.1f} qps",
        ),
    ]

    for metric, base_v, fine_v, delta in rows:
        print(f"{metric:<30} | {base_v:<16} | {fine_v:<16} | {delta:<12}")

    print("=" * 80 + "\n")


def main():
    args = parse_args()
    device = get_default_device(args.device)

    print("=== Laya Benchmark Evaluation Runner ===")
    print(f"Device: {device}")

    # 1. Load benchmark dataset
    cases, source_desc = load_benchmark_cases(args)
    print(f"Loaded {len(cases)} benchmark cases from {source_desc}.")
    if not cases:
        print("Error: No benchmark cases found.")
        sys.exit(1)

    # 2. Evaluate target model
    print(f"\n[1/2] Loading target model: {args.model_path}...")
    target_agent = Agent(model_id_or_path=args.model_path, device=device)
    print(f"Running benchmark inference on {args.model_path}...")
    target_metrics = evaluate_model(target_agent, cases, model_name=args.model_path)

    base_metrics = None
    if args.compare_base:
        # 3. Evaluate base model
        print(f"\n[2/2] Loading base model for comparison: {args.base_model_path}...")
        base_agent = Agent(model_id_or_path=args.base_model_path, device=device)
        print(f"Running benchmark inference on {args.base_model_path}...")
        base_metrics = evaluate_model(base_agent, cases, model_name=args.base_model_path)

        # Print comparison table
        print_comparison_table(base_metrics, target_metrics)
    else:
        # Print single model summary
        print_single_summary(target_metrics)

    # 4. Optional export to JSON
    if args.output_json:
        export_data = {
            "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "device": device,
            "benchmark_source": source_desc,
            "target_model": target_metrics,
        }
        if base_metrics:
            export_data["base_model"] = base_metrics
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)
        print(f"Saved complete evaluation report to: {args.output_json}")


if __name__ == "__main__":
    main()
