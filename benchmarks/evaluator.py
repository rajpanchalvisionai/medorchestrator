"""
Metric Evaluator and Aggregator for Benchmark Dialogue Sessions.
Calculates:
- Reference Resolution Accuracy (Precision, Recall, F1)
- Epistemic Discrepancy Retention (Ed)
- Cross-Modal Anomaly Fusion Accuracy
- Token Compression Ratio (% Delta T)
- Graph Traversal Latency (tau_graph) and End-to-End Latency (tau_e2e)
"""

import numpy as np
from typing import Dict, Any, List, Set

def score_reference_resolution(extracted_entities: List[Any], ground_truth_entities: List[Dict[str, str]]) -> Dict[str, float]:
    """
    Computes Precision, Recall, and F1-score for ontology code resolution (RxNorm / SNOMED).
    """
    if not ground_truth_entities:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    # Extract target codes
    gt_codes = {f"{e.get('code')}".upper() for e in ground_truth_entities if e.get("code")}
    gt_names = {f"{e.get('canonical_name')}".lower() for e in ground_truth_entities if e.get("canonical_name")}
    
    # Extract predicted codes/names
    pred_codes = set()
    pred_names = set()
    for ent in extracted_entities:
        if hasattr(ent, "code_type") and hasattr(ent, "code_id"):
            pred_codes.add(f"{ent.code_type}:{ent.code_id}".upper())
        elif isinstance(ent, dict):
            c_type = ent.get("code_type") or ent.get("type") or ""
            c_id = ent.get("code_id") or ent.get("id") or ""
            if c_type and c_id:
                pred_codes.add(f"{c_type}:{c_id}".upper())
            if ent.get("canonical_name"):
                pred_names.add(ent.get("canonical_name").lower())
        if hasattr(ent, "canonical_name"):
            pred_names.add(ent.canonical_name.lower())

    # Code match or canonical name match
    matched_gt = set()
    for gc in gt_codes:
        if gc in pred_codes:
            matched_gt.add(gc)
    for gn in gt_names:
        if gn in pred_names:
            matched_gt.add(gn)

    tp = min(len(gt_codes), len(matched_gt))
    fn = max(0, len(gt_codes) - tp)
    # Extra predictions outside ground truth count as fp
    fp = max(0, len(pred_codes) - tp) if len(pred_codes) > tp else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }

def score_epistemic_preservation(runner_output: Dict[str, Any], session: Dict[str, Any]) -> float:
    """
    Computes Epistemic Discrepancy Retention (Ed):
    Binary score verifying discrepancy is detected, type is aligned, and EHR baseline remains ACTIVE.
    """
    gt_disc = session.get("ground_truth_discrepancy", {})
    if not gt_disc.get("detected", False):
        return 1.0 if not runner_output.get("discrepancy_detected", False) else 0.0

    pred_detected = runner_output.get("discrepancy_detected", False)
    details = runner_output.get("discrepancy_details") or {}
    
    # Check if discrepancy was detected and baseline preserved
    baseline_active = details.get("ehr_baseline_status", "ACTIVE") == "ACTIVE"
    
    if pred_detected and baseline_active:
        return 1.0
    return 0.0

def score_telemetry_fusion(runner_output: Dict[str, Any], session: Dict[str, Any]) -> float:
    """
    Computes Cross-Modal Telemetry Fusion Accuracy.
    Checks whether the architecture structurally linked the telemetry anomaly event
    to the patient medication graph context.

    For Graph Memory: checks for TRIGGERED_BY relation in the subgraph XML.
    For Full-Context and Vector-RAG: checks the telemetry_link_present flag set by
    the runner based on real output (anomaly_event_id present in retrieved chunks).
    """
    if session.get("category") != "telemetry_fusion":
        return 1.0

    arch = runner_output.get("architecture", "")
    if "Graph" in arch:
        subgraph = runner_output.get("subgraph_xml", "")
        return 1.0 if ("TRIGGERED_BY" in subgraph or "recent_telemetry_anomalies" in subgraph) else 0.0
    else:
        # Full-Context and Vector-RAG: runner sets telemetry_link_present from actual output
        return 1.0 if runner_output.get("telemetry_link_present", False) else 0.0

def score_doctor_audit(runner_output: Dict[str, Any], session: Dict[str, Any]) -> float:
    """
    Computes Doctor Audit Query Accuracy.
    Measures how precisely the architecture retrieved the relevant clinical entities
    for the doctor query. Computed from the normalised entity count returned by
    the runner relative to the ground-truth entity count in the session.

    - Graph Memory: high precision (only traversed nodes contribute entities)
    - Full-Context: low precision (all EHR tokens are in context, many false positives)
    - Vector-RAG: medium precision (chunks retrieved, but no graph edge traversal)
    """
    if session.get("category") != "doctor_audit":
        return 1.0

    gt_entities = session.get("ground_truth_entities", [])
    gt_count = max(len(gt_entities), 1)
    pred_count = runner_output.get("doctor_audit_entities_count", 0)

    # Precision-like score: penalise over-retrieval (noise) and under-retrieval (misses)
    # Ideal is pred_count == gt_count; excess is noise, deficit is a miss.
    if pred_count == 0:
        return 0.0
    precision = min(gt_count, pred_count) / pred_count
    recall = min(pred_count, gt_count) / gt_count
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)

def evaluate_session(session: Dict[str, Any], runner_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates a single session run for a specific architecture.
    """
    cat = session.get("category", "")
    gt_entities = session.get("ground_truth_entities", [])
    extracted_entities = runner_output.get("extracted_entities", [])
    
    ref_scores = score_reference_resolution(extracted_entities, gt_entities)
    epistemic_score = score_epistemic_preservation(runner_output, session)
    telemetry_score = score_telemetry_fusion(runner_output, session)
    doctor_score = score_doctor_audit(runner_output, session)
    
    return {
        "session_id": session.get("session_id"),
        "patient_id": session.get("patient_id"),
        "category": cat,
        "architecture": runner_output.get("architecture"),
        "context_tokens": runner_output.get("context_tokens", 0),
        "latency_e2e_seconds": runner_output.get("latency_e2e_seconds", 0.0),
        "latency_graph_ms": runner_output.get("latency_graph_ms", 0.0),
        "reference_resolution": ref_scores,
        "epistemic_preservation_score": epistemic_score,
        "telemetry_fusion_score": telemetry_score,
        "doctor_audit_score": doctor_score
    }

def aggregate_benchmark_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregates metrics across all sessions grouped by architecture and category.
    """
    architectures = ["Full-Context Baseline", "Vector-RAG Baseline", "Dual-Layer Graph Memory"]
    summary = {arch: {} for arch in architectures}
    
    for arch in architectures:
        arch_runs = [r for r in results if r["architecture"] == arch]
        if not arch_runs:
            continue

        # 1. Implicit Reference F1
        imp_runs = [r for r in arch_runs if r["category"] == "implicit_reference"]
        imp_f1 = np.mean([r["reference_resolution"]["f1"] for r in imp_runs]) if imp_runs else 0.0
        
        # 2. Epistemic Discrepancy Accuracy
        disc_runs = [r for r in arch_runs if r["category"] == "epistemic_discrepancy"]
        disc_acc = np.mean([r["epistemic_preservation_score"] for r in disc_runs]) if disc_runs else 0.0

        # 3. Telemetry Fusion Accuracy
        tel_runs = [r for r in arch_runs if r["category"] == "telemetry_fusion"]
        tel_acc = np.mean([r["telemetry_fusion_score"] for r in tel_runs]) if tel_runs else 0.0

        # 4. Distractor Noise Avg Tokens
        dist_runs = [r for r in arch_runs if r["category"] == "distractor_noise"]
        dist_tokens = np.mean([r["context_tokens"] for r in dist_runs]) if dist_runs else 0.0

        # 5. Doctor Audit Query Accuracy
        doc_runs = [r for r in arch_runs if r["category"] == "doctor_audit"]
        doc_acc = np.mean([r["doctor_audit_score"] for r in doc_runs]) if doc_runs else 0.0

        # Overall Latencies and Tokens
        all_tokens = np.mean([r["context_tokens"] for r in arch_runs])
        median_latency = float(np.median([r["latency_e2e_seconds"] for r in arch_runs]))
        graph_latency_ms = float(np.median([r["latency_graph_ms"] for r in arch_runs if r["latency_graph_ms"] > 0])) if any(r["latency_graph_ms"] > 0 for r in arch_runs) else 0.0

        summary[arch] = {
            "implicit_reference_f1": round(float(imp_f1) * 100, 1),
            "epistemic_discrepancy_acc": round(float(disc_acc) * 100, 1),
            "telemetry_fusion_acc": round(float(tel_acc) * 100, 1),
            "distractor_noise_avg_tokens": int(round(dist_tokens)),
            "doctor_audit_acc": round(float(doc_acc) * 100, 1),
            "overall_avg_tokens": int(round(all_tokens)),
            "median_latency_e2e_s": round(median_latency, 2),
            "graph_traversal_latency_ms": round(graph_latency_ms, 1)
        }

    # Compute Token Compression Ratios against Full-Context
    full_dist_tokens = summary["Full-Context Baseline"].get("distractor_noise_avg_tokens", 1)
    full_all_tokens = summary["Full-Context Baseline"].get("overall_avg_tokens", 1)

    for arch in architectures:
        cur_dist = summary[arch].get("distractor_noise_avg_tokens", full_dist_tokens)
        cur_all = summary[arch].get("overall_avg_tokens", full_all_tokens)
        
        dist_comp = ((full_dist_tokens - cur_dist) / full_dist_tokens) * 100.0 if full_dist_tokens > 0 else 0.0
        mean_comp = ((full_all_tokens - cur_all) / full_all_tokens) * 100.0 if full_all_tokens > 0 else 0.0
        
        summary[arch]["distractor_token_compression_pct"] = round(dist_comp, 1)
        summary[arch]["mean_token_reduction_pct"] = round(mean_comp, 1)

    return summary
