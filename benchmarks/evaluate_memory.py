"""
Dual-Layer Graph Memory vs Baseline Evaluation Harness.
Loads MMASH Grounded Evaluation Suite, runs across three architectures,
computes metrics, saves JSON summary incrementally, handles API rate limits gracefully,
and supports resuming from checkpoints.
"""

import os
import sys
import json
import argparse
import traceback
from typing import List, Dict, Any, Set

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.runners import (
    run_full_context_baseline,
    run_vector_rag_baseline,
    run_dual_layer_graph
)
from benchmarks.evaluator import (
    evaluate_session,
    aggregate_benchmark_results
)


def _save_checkpoint(output_path: str, evaluated_records: List[Dict[str, Any]], total_sessions: int) -> Dict[str, Any]:
    """Computes aggregate metrics and atomically saves current progress to JSON."""
    summary = aggregate_benchmark_results(evaluated_records) if evaluated_records else {}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp_path = output_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_sessions_in_dataset": total_sessions,
            "sessions_completed": len({r.get("session_id") for r in evaluated_records}),
            "summary_metrics": summary,
            "session_evaluations": evaluated_records
        }, f, indent=2)
    
    # Atomic replace
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except OSError:
            pass
    os.replace(temp_path, output_path)
    return summary


def run_evaluation(
    dataset_path: str = "benchmarks/dataset/mmash_grounded_eval_suite.json",
    output_path: str = "benchmarks/results/benchmark_summary.json",
    limit: int = None,
    resume: bool = True
) -> Dict[str, Any]:
    """
    Executes the full evaluation suite across all benchmark sessions with
    incremental saving and automatic resume capabilities.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        sessions: List[Dict[str, Any]] = json.load(f)

    if limit is not None and limit > 0:
        sessions = sessions[:limit]

    evaluated_records: List[Dict[str, Any]] = []
    completed_session_ids: Set[str] = set()

    # If resuming, load previously completed sessions from output_path
    if resume and os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
                prev_records = prev_data.get("session_evaluations", [])
                
                # Check which sessions have all 3 architectures completed
                arch_count_by_session: Dict[str, int] = {}
                for rec in prev_records:
                    sid = rec.get("session_id")
                    if sid:
                        arch_count_by_session[sid] = arch_count_by_session.get(sid, 0) + 1

                for sid, count in arch_count_by_session.items():
                    if count >= 3:
                        completed_session_ids.add(sid)

                # Keep only valid records for completed sessions
                evaluated_records = [r for r in prev_records if r.get("session_id") in completed_session_ids]
                
                if completed_session_ids:
                    print(f"[RESUME] Found existing checkpoint at {output_path} with {len(completed_session_ids)} completed sessions. Skipping them.")
        except Exception as err:
            print(f"[WARNING] Could not load previous checkpoint ({err}). Starting fresh.")
            evaluated_records = []
            completed_session_ids = set()

    print(f"Loaded {len(sessions)} total sessions. Remaining to evaluate: {len(sessions) - len(completed_session_ids)}...")

    interrupted = False

    for idx, session in enumerate(sessions, start=1):
        sess_id = session.get("session_id", f"sess_{idx}")
        cat = session.get("category", "unknown")

        if sess_id in completed_session_ids:
            print(f"[{idx}/{len(sessions)}] Skipping {sess_id} (already completed in checkpoint)", flush=True)
            continue

        print(f"[{idx}/{len(sessions)}] Evaluating {sess_id} ({cat})...", flush=True)

        session_records = []
        try:
            # 1. Full-Context Baseline
            fc_out = run_full_context_baseline(session)
            session_records.append(evaluate_session(session, fc_out))

            # 2. Vector-RAG Baseline
            vr_out = run_vector_rag_baseline(session)
            session_records.append(evaluate_session(session, vr_out))

            # 3. Dual-Layer Graph Memory
            gm_out = run_dual_layer_graph(session)
            session_records.append(evaluate_session(session, gm_out))

            # Commit session records
            evaluated_records.extend(session_records)
            completed_session_ids.add(sess_id)

            # Save checkpoint incrementally after every session
            _save_checkpoint(output_path, evaluated_records, len(sessions))

        except (KeyboardInterrupt, Exception) as exc:
            interrupted = True
            is_quota = any(kw in str(exc).lower() for kw in ["quota", "resourceexhausted", "429", "rate limit", "exceeded"])
            
            print("\n" + "!" * 80)
            if is_quota:
                print(f"[API QUOTA / RATE LIMIT REACHED] at session {idx}/{len(sessions)} ({sess_id})")
                print(f"Details: {exc}")
                print(f"\nAll {len(completed_session_ids)} completed sessions have been securely saved to {output_path}.")
                print("You can safely resume anytime tomorrow or after quota resets by running:")
                print("  python benchmarks/evaluate_memory.py --resume")
            elif isinstance(exc, KeyboardInterrupt):
                print(f"\n[INTERRUPTED BY USER] Execution paused at session {idx}/{len(sessions)}.")
                print(f"All {len(completed_session_ids)} completed sessions saved to {output_path}.")
                print("To resume later, run:")
                print("  python benchmarks/evaluate_memory.py --resume")
            else:
                print(f"[ERROR OCCURRED] at session {idx}/{len(sessions)} ({sess_id}): {exc}")
                traceback.print_exc()
                print(f"\nAll {len(completed_session_ids)} completed sessions saved to {output_path}.")
            print("!" * 80 + "\n")
            break

    # Final summary update
    summary = _save_checkpoint(output_path, evaluated_records, len(sessions))
    print(f"\nSaved evaluation summary to {output_path} ({len(completed_session_ids)}/{len(sessions)} sessions completed)")

    if interrupted:
        print("\nNote: Benchmark was paused before all sessions completed. Partial results shown below:")

    return summary


def format_summary_table(summary: Dict[str, Any]) -> str:
    """Formats aggregated metrics into publication-ready comparative table."""
    fc = summary.get("Full-Context Baseline", {})
    vr = summary.get("Vector-RAG Baseline", {})
    gm = summary.get("Dual-Layer Graph Memory", {})

    lines = [
        "=" * 90,
        "DUAL-LAYER GRAPH MEMORY BENCHMARK EVALUATION RESULTS",
        "=" * 90,
        f"{'Category':<22} {'Metric':<22} {'Full-Context':<15} {'Vector-RAG':<15} {'Graph Memory':<15}",
        "-" * 90,
        f"{'Implicit Reference':<22} {'Reference F1-Score':<22} {fc.get('implicit_reference_f1', 0.0):.1f}%{'':<10} {vr.get('implicit_reference_f1', 0.0):.1f}%{'':<10} {gm.get('implicit_reference_f1', 0.0):.1f}%",
        f"{'Epistemic Disc.':<22} {'Discrepancy Accuracy':<22} {fc.get('epistemic_discrepancy_acc', 0.0):.1f}%{'':<10} {vr.get('epistemic_discrepancy_acc', 0.0):.1f}%{'':<10} {gm.get('epistemic_discrepancy_acc', 0.0):.1f}%",
        f"{'Telemetry Fusion':<22} {'Cross-Modal Accuracy':<22} {fc.get('telemetry_fusion_acc', 0.0):.1f}%{'':<10} {vr.get('telemetry_fusion_acc', 0.0):.1f}%{'':<10} {gm.get('telemetry_fusion_acc', 0.0):.1f}%",
        f"{'Distractor Noise':<22} {'Avg Context Tokens':<22} {fc.get('distractor_noise_avg_tokens', 0):,d} tokens{'':<3} {vr.get('distractor_noise_avg_tokens', 0):,d} tokens{'':<5} {gm.get('distractor_noise_avg_tokens', 0):,d} tokens",
        f"{'Distractor Noise':<22} {'Token Compression':<22} {'0.0% (Base)':<15} {vr.get('distractor_token_compression_pct', 0.0):.1f}%{'':<10} {gm.get('distractor_token_compression_pct', 0.0):.1f}%",
        f"{'Doctor Audit':<22} {'Query Accuracy':<22} {fc.get('doctor_audit_acc', 0.0):.1f}%{'':<10} {vr.get('doctor_audit_acc', 0.0):.1f}%{'':<10} {gm.get('doctor_audit_acc', 0.0):.1f}%",
        "-" * 90,
        f"{'OVERALL PERFORMANCE':<22} {'Median Latency (e2e)':<22} {fc.get('median_latency_e2e_s', 0.0):.2f}s{'':<10} {vr.get('median_latency_e2e_s', 0.0):.2f}s{'':<10} {gm.get('median_latency_e2e_s', 0.0):.2f}s",
        f"{'':<22} {'Mean Token Reduction':<22} {'--':<15} {vr.get('mean_token_reduction_pct', 0.0):.1f}%{'':<10} {gm.get('mean_token_reduction_pct', 0.0):.1f}%",
        "=" * 90
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Memory Architectures on Grounded Clinical Dialogue Benchmark")
    parser.add_argument("--dataset", default="benchmarks/dataset/mmash_grounded_eval_suite.json", help="Path to grounded dataset JSON")
    parser.add_argument("--output", default="benchmarks/results/benchmark_summary.json", help="Output summary JSON path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of sessions to evaluate for testing")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from existing checkpoint if available (default: True)")
    parser.add_argument("--force", action="store_true", help="Force restart evaluation from session 1 ignoring previous checkpoint")
    args = parser.parse_args()

    resume_flag = args.resume and not args.force

    summary = run_evaluation(
        dataset_path=args.dataset,
        output_path=args.output,
        limit=args.limit,
        resume=resume_flag
    )
    if summary:
        print("\n" + format_summary_table(summary))


if __name__ == "__main__":
    main()
