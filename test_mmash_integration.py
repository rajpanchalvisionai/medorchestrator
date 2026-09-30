import os
import pytest
from datetime import datetime
from engine.mmash_loader import MMASHDatasetLoader
from engine.ingestion import IngestionRouter
from schemas.inputs import TelemetryPayload, PatientReportPayload, DoctorQueryPayload

USER_1_DIR = os.path.join(os.path.dirname(__file__), "DataPaper", "user_1")

def test_mmash_loader_structure():
    """Verify that MMASH user_1 contains expected files and parses correctly."""
    assert os.path.isdir(USER_1_DIR), f"Directory {USER_1_DIR} does not exist"
    
    loader = MMASHDatasetLoader()
    
    # 1. Test telemetry stream
    telemetry_list = list(loader.stream_telemetry(USER_1_DIR, limit=10))
    assert len(telemetry_list) == 10
    
    for t in telemetry_list:
        assert isinstance(t, TelemetryPayload)
        assert t.patient_id == "user_1"
        assert t.heart_rate > 0
        assert t.hrv_rmssd > 0
        assert t.accelerometer_magnitude >= 0
        assert t.channel_source == "wearable_stream"

    # 2. Test patient reports
    reports = loader.load_patient_reports(USER_1_DIR)
    assert len(reports) > 0
    for r in reports:
        assert isinstance(r, PatientReportPayload)
        assert r.patient_id == "user_1"
        assert 1 <= r.perceived_severity <= 10
        assert len(r.raw_text) > 0
        assert r.channel_source == "patient_chat"

    # 3. Test doctor queries
    queries = loader.generate_doctor_queries(USER_1_DIR)
    assert len(queries) >= 2
    for q in queries:
        assert isinstance(q, DoctorQueryPayload)
        assert q.patient_id == "user_1"
        assert "dr_" in q.doctor_id
        assert len(q.query_text) > 0
        assert q.channel_source == "doctor_portal"

def test_mmash_normal_telemetry_buffering():
    """Verify that normal resting telemetry from user_1 passes quietly without waking LLM."""
    loader = MMASHDatasetLoader()
    router = IngestionRouter()
    
    # Rows 0-10 in user_1 are normal baseline (HR ~ 68 BPM, Accel moderate)
    telemetry_samples = list(loader.stream_telemetry(USER_1_DIR, limit=10, start_row=0))
    
    events = [router.process_event(t) for t in telemetry_samples]
    
    # Check that normal readings did not trigger LLM processing
    for ev in events:
        assert ev.requires_llm_processing is False
        assert ev.channel == "wearable_anomaly"
        assert ev.patient_id == "user_1"
        
    # Check that normal readings were buffered locally
    assert len(router.local_normal_buffer) == 10

def test_mmash_tachycardia_anomaly_detection():
    """Verify that real sustained tachycardia in user_1 (HR > 120, low motion) triggers anomaly."""
    loader = MMASHDatasetLoader()
    router = IngestionRouter()
    
    # In user_1, row 16 has HR=133, Accel=0.0.
    # We stream rows 0 through 25 to let the 10-second rolling window evaluate
    detected_anomaly = False
    for t in loader.stream_telemetry(USER_1_DIR, limit=30, start_row=0):
        event = router.process_event(t)
        if event.requires_llm_processing:
            detected_anomaly = True
            assert event.channel == "wearable_anomaly"
            assert event.payload.heart_rate > 120.0
            break

    assert detected_anomaly, "Failed to detect genuine tachycardia anomaly in user_1 telemetry stream!"

def test_mmash_patient_and_doctor_channel_routing():
    """Verify that real patient self-reports and doctor queries route to requires_llm_processing=True."""
    loader = MMASHDatasetLoader()
    router = IngestionRouter()

    # Channel 2: Patient reports
    reports = loader.load_patient_reports(USER_1_DIR)
    for r in reports[:5]:
        event = router.process_event(r)
        assert event.channel == "patient_chat"
        assert event.requires_llm_processing is True
        assert event.patient_id == "user_1"

    # Channel 3: Doctor queries
    queries = loader.generate_doctor_queries(USER_1_DIR)
    for q in queries:
        event = router.process_event(q)
        assert event.channel == "doctor_portal"
        assert event.requires_llm_processing is True
        assert event.patient_id == "user_1"

if __name__ == "__main__":
    print("Running MMASH integration tests...")
    test_mmash_loader_structure()
    print("✓ MMASH loader structure test passed.")
    test_mmash_normal_telemetry_buffering()
    print("✓ MMASH normal telemetry buffering test passed.")
    test_mmash_tachycardia_anomaly_detection()
    print("✓ MMASH tachycardia anomaly detection test passed.")
    test_mmash_patient_and_doctor_channel_routing()
    print("✓ MMASH multi-channel routing test passed.")
    print("\nAll MMASH integration tests PASSED successfully!")
