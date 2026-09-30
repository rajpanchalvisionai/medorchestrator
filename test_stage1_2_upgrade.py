import time
import pytest
from datetime import datetime, timedelta

from schemas.inputs import TelemetryPayload, PatientReportPayload, DoctorQueryPayload
from engine.signal_processor import PolarsTelemetryBuffer, GatingResult
from engine.normalizer import ClinicalTextNormalizer
from engine.ingestion import IngestionRouter

def test_polars_throughput_benchmark():
    """
    Benchmark: Ingestion of 1,000 telemetry samples through PolarsTelemetryBuffer
    must complete in < 10ms.
    """
    buffer = PolarsTelemetryBuffer(window_seconds=10.0)
    base_time = datetime(2026, 8, 23, 12, 0, 0)

    # Prepare 1,000 telemetry samples (simulating a high-frequency telemetry stream)
    samples = [
        TelemetryPayload(
            patient_id="benchmark_patient",
            timestamp=base_time + timedelta(milliseconds=10 * i),
            heart_rate=78.0 + (i % 5),
            hrv_rmssd=52.0,
            accelerometer_magnitude=0.45,
            channel_source="wearable_stream"
        )
        for i in range(1000)
    ]

    # Benchmark row-by-row streaming ingestion
    start_time = time.perf_counter()
    for sample in samples:
        buffer.append(sample)
    duration_ms = (time.perf_counter() - start_time) * 1000.0

    print(f"\n[BENCHMARK] Ingested 1,000 telemetry samples in {duration_ms:.3f} ms")
    assert duration_ms < 10.0, f"Polars ingestion exceeded 10ms threshold: {duration_ms:.3f} ms"
    assert len(buffer.df) > 0

def test_gating_precision():
    """
    Gating Precision: A simulated sequence of HR=130 BPM, Accel=0.05 g over 10 seconds
    accurately sets H(W_t) = 1.0 and triggers an anomaly event.
    """
    router = IngestionRouter(accel_threshold=0.1, nominal_baseline_hr=75.0)
    patient_id = "patient_tachy_01"
    base_time = datetime(2026, 8, 23, 14, 0, 0)

    # 1. Normal baseline (10 seconds): HR=72 BPM, Accel=0.05 g
    for i in range(10):
        normal_payload = TelemetryPayload(
            patient_id=patient_id,
            timestamp=base_time + timedelta(seconds=i),
            heart_rate=72.0,
            hrv_rmssd=55.0,
            accelerometer_magnitude=0.05,
            channel_source="wearable_stream"
        )
        event = router.process_event(normal_payload)
        assert event.requires_llm_processing is False
        assert event.entropy_score == 0.0

    # 2. Simulated resting tachycardia sequence: HR=130 BPM, Accel=0.05 g over 10 seconds
    anomaly_events = []
    for i in range(10, 22):
        tachy_payload = TelemetryPayload(
            patient_id=patient_id,
            timestamp=base_time + timedelta(seconds=i),
            heart_rate=130.0,
            hrv_rmssd=38.0,
            accelerometer_magnitude=0.05,
            channel_source="wearable_stream"
        )
        event = router.process_event(tachy_payload)
        anomaly_events.append(event)

    # The later events (once rolling window is dominated by 130 BPM for >= 10s) must trigger H(W_t) = 1.0
    latest_event = anomaly_events[-1]
    assert latest_event.requires_llm_processing is True
    assert latest_event.entropy_score == 1.0
    assert latest_event.baseline_diff == 55.0  # 130.0 - 75.0
    assert latest_event.channel == "wearable_anomaly"

def test_entity_resolution_accuracy():
    """
    Entity Resolution Accuracy: Passing the text:
    'I took my blood pressure pill and felt dizzy'
    returns normalized entities containing RxNorm and SNOMED mappings.
    """
    normalizer = ClinicalTextNormalizer()
    text = "I took my blood pressure pill and felt dizzy"
    entities = normalizer.normalize(text)

    assert len(entities) >= 2

    # Verify RxNorm mapping for 'blood pressure pill'
    rxnorm_matches = [e for e in entities if e.code_type == "RxNorm"]
    assert len(rxnorm_matches) >= 1
    rxnorm_entity = rxnorm_matches[0]
    assert rxnorm_entity.code_id == "6918"
    assert rxnorm_entity.canonical_name == "Metoprolol"
    assert rxnorm_entity.confidence >= 0.90

    # Verify SNOMED mapping for 'dizzy'
    snomed_matches = [e for e in entities if e.code_type == "SNOMED"]
    assert len(snomed_matches) >= 1
    snomed_entity = snomed_matches[0]
    assert snomed_entity.code_id == "25064002"
    assert snomed_entity.canonical_name == "Dizziness"
    assert snomed_entity.confidence >= 0.90

def test_integrated_router_channels_2_and_3():
    """
    Verify that Channel 2 (Patient Chat) and Channel 3 (Doctor Query)
    run through normalizer and enrich normalized_entities.
    """
    router = IngestionRouter()
    base_time = datetime(2026, 8, 23, 15, 30, 0)

    # Channel 2: Patient chat with medications and symptoms
    patient_report = PatientReportPayload(
        patient_id="patient_chat_01",
        timestamp=base_time,
        raw_text="I took my blood pressure pill and felt dizzy and lightheaded after my morning pill.",
        perceived_severity=7,
        channel_source="patient_chat"
    )
    event_report = router.process_event(patient_report)
    assert event_report.channel == "patient_chat"
    assert event_report.requires_llm_processing is True
    assert len(event_report.payload.normalized_entities) >= 2
    code_types = {e.code_type for e in event_report.payload.normalized_entities}
    assert "RxNorm" in code_types
    assert "SNOMED" in code_types

    # Channel 3: Doctor query with clinical concepts
    doctor_query = DoctorQueryPayload(
        doctor_id="dr_smith",
        patient_id="patient_chat_01",
        timestamp=base_time,
        query_text="Audit patient for tachycardia spikes and monitor heart rate variability during rest.",
        channel_source="doctor_portal"
    )
    event_query = router.process_event(doctor_query)
    assert event_query.channel == "doctor_portal"
    assert event_query.requires_llm_processing is True
    assert len(event_query.payload.normalized_entities) >= 1
    concepts = [e.canonical_name for e in event_query.payload.normalized_entities]
    assert "Tachycardia" in concepts or "Heart Rate Variability" in concepts
