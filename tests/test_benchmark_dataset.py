"""
Regression and Integrity Test Suite for MMASH Grounded Benchmark Dataset.
Validates structural schema, clinical entity grounding, category quotas,
and physiological telemetry alignment.
"""

import os
import json
import pytest

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "dataset", "mmash_grounded_eval_suite.json")

@pytest.fixture(scope="session")
def dataset():
    assert os.path.exists(DATASET_PATH), f"Benchmark dataset file not found at {DATASET_PATH}"
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def test_dataset_total_sessions_count(dataset):
    assert isinstance(dataset, list)
    assert len(dataset) == 66, f"Expected 66 benchmark session objects, found {len(dataset)}"

def test_dataset_category_distribution(dataset):
    counts = {}
    for s in dataset:
        cat = s.get("category")
        counts[cat] = counts.get(cat, 0) + 1

    assert counts.get("implicit_reference") == 20
    assert counts.get("epistemic_discrepancy") == 16
    assert counts.get("telemetry_fusion") == 13
    assert counts.get("distractor_noise") == 10
    assert counts.get("doctor_audit") == 7

def test_dataset_user_coverage(dataset):
    user_counts = {}
    for s in dataset:
        u = s.get("patient_id")
        user_counts[u] = user_counts.get(u, 0) + 1

    expected_users = {f"user_{i}" for i in range(1, 23)}
    assert set(user_counts.keys()) == expected_users
    for u, count in user_counts.items():
        assert count == 3, f"User {u} has {count} sessions instead of 3"

def test_dataset_session_schema_integrity(dataset):
    for s in dataset:
        assert "session_id" in s and s["session_id"]
        assert "patient_id" in s and s["patient_id"].startswith("user_")
        assert "category" in s
        assert "ehr_baseline" in s
        assert "ground_truth_entities" in s
        assert "ground_truth_discrepancy" in s
        assert "turns" in s and len(s["turns"]) >= 2

        # Check turn formatting
        for idx, turn in enumerate(s["turns"], start=1):
            assert turn["turn_index"] == idx
            assert turn["channel"] in {"patient_chat", "doctor_portal", "wearable_anomaly"}
            assert turn["speaker"] in {"patient", "agent", "doctor", "wearable_stream"}
            assert isinstance(turn["text"], str) and len(turn["text"].strip()) > 0
            assert "timestamp" in turn

        # Check EHR baseline
        ehr = s["ehr_baseline"]
        assert "demographics" in ehr and ehr["demographics"]["age"] > 0
        assert "conditions" in ehr and len(ehr["conditions"]) >= 1
        assert "prescriptions" in ehr and len(ehr["prescriptions"]) >= 1
        assert "threshold_protocols" in ehr
        assert "psychophysiological_baseline" in ehr

        # Check entity codes
        for ent in s["ground_truth_entities"]:
            assert "raw_text" in ent and ent["raw_text"]
            assert "canonical_name" in ent and ent["canonical_name"]
            assert "code" in ent
            assert ent["code"].startswith("RxNorm:") or ent["code"].startswith("SNOMED:")

def test_epistemic_discrepancy_grounding(dataset):
    discrepancy_sessions = [s for s in dataset if s["category"] == "epistemic_discrepancy"]
    assert len(discrepancy_sessions) == 16
    for s in discrepancy_sessions:
        disc = s["ground_truth_discrepancy"]
        assert disc["detected"] is True
        assert disc["type"] in {"UNVERIFIED_ADHERENCE_STOP", "UNVERIFIED_DOSAGE_CHANGE"}
        assert disc["ehr_baseline_status"] == "ACTIVE"
        assert "target_entity" in disc

def test_telemetry_fusion_grounding(dataset):
    fusion_sessions = [s for s in dataset if s["category"] == "telemetry_fusion"]
    assert len(fusion_sessions) == 13
    for s in fusion_sessions:
        assert "ground_truth_telemetry_link" in s
        link = s["ground_truth_telemetry_link"]
        assert link["relation"] == "TRIGGERED_BY"
        assert link["heart_rate"] >= 115.0
        assert link["accelerometer_magnitude"] <= 0.1
        assert "anomaly_event_id" in link

def test_distractor_noise_grounding(dataset):
    noise_sessions = [s for s in dataset if s["category"] == "distractor_noise"]
    assert len(noise_sessions) == 10
    for s in noise_sessions:
        assert len(s["turns"]) >= 8, f"Distractor noise session should have 8-12 turns, got {len(s['turns'])}"
        assert "subgraph_evaluation_targets" in s
        targets = s["subgraph_evaluation_targets"]
        assert targets["max_target_context_tokens"] <= 400
        assert targets["expected_pruned_turns_count"] >= 6

def test_doctor_audit_grounding(dataset):
    audit_sessions = [s for s in dataset if s["category"] == "doctor_audit"]
    assert len(audit_sessions) == 7
    for s in audit_sessions:
        assert s["turns"][0]["channel"] == "doctor_portal"
        assert s["turns"][0]["speaker"] == "doctor"
        assert "ground_truth_audit_targets" in s
