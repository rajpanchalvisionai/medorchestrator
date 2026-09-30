"""
Unit and Integration Tests for Memory Benchmark Evaluation Harness.
Tests all three architectural runners and metric evaluation over session slices.
"""

import os
import json
import pytest

from benchmarks.runners import (
    run_full_context_baseline,
    run_vector_rag_baseline,
    run_dual_layer_graph
)
from benchmarks.evaluator import (
    evaluate_session,
    aggregate_benchmark_results,
    score_reference_resolution,
    score_epistemic_preservation
)
from benchmarks.evaluate_memory import run_evaluation

SAMPLE_DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "dataset", "mmash_grounded_eval_suite.json")
OUTPUT_SUMMARY_PATH = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "results", "test_harness_summary.json")

@pytest.fixture
def sample_sessions():
    assert os.path.exists(SAMPLE_DATASET_PATH), "Benchmark dataset must exist"
    with open(SAMPLE_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[:2]

def test_full_context_runner_execution(sample_sessions):
    session = sample_sessions[0]
    result = run_full_context_baseline(session)
    assert result["architecture"] == "Full-Context Baseline"
    assert result["context_tokens"] > 0
    assert result["latency_e2e_seconds"] > 0
    assert isinstance(result["extracted_entities"], list)

def test_vector_rag_runner_execution(sample_sessions):
    session = sample_sessions[0]
    result = run_vector_rag_baseline(session)
    assert result["architecture"] == "Vector-RAG Baseline"
    assert result["context_tokens"] > 0
    assert result["latency_e2e_seconds"] > 0
    assert "retrieved_chunks_count" in result

def test_dual_layer_graph_runner_execution(sample_sessions):
    session = sample_sessions[0]
    result = run_dual_layer_graph(session)
    assert result["architecture"] == "Dual-Layer Graph Memory"
    assert result["context_tokens"] > 0
    assert result["latency_graph_ms"] > 0
    assert "<patient_profile>" in result["subgraph_xml"]

def test_metric_evaluator_scoring(sample_sessions):
    session = sample_sessions[0]
    gm_out = run_dual_layer_graph(session)
    eval_record = evaluate_session(session, gm_out)
    
    assert eval_record["session_id"] == session["session_id"]
    assert "reference_resolution" in eval_record
    assert "f1" in eval_record["reference_resolution"]
    assert "epistemic_preservation_score" in eval_record
    assert eval_record["context_tokens"] > 0

def test_evaluate_memory_harness_slice_execution(sample_sessions):
    summary = run_evaluation(
        dataset_path=SAMPLE_DATASET_PATH,
        output_path=OUTPUT_SUMMARY_PATH,
        limit=2
    )
    assert os.path.exists(OUTPUT_SUMMARY_PATH)
    assert "Full-Context Baseline" in summary
    assert "Vector-RAG Baseline" in summary
    assert "Dual-Layer Graph Memory" in summary
    
    with open(OUTPUT_SUMMARY_PATH, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
    assert saved_data["total_sessions_evaluated"] == 2
    assert len(saved_data["session_evaluations"]) == 6  # 2 sessions * 3 architectures
