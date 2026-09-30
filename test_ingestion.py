import asyncio
from datetime import datetime, timedelta
from schemas.inputs import (
    TelemetryPayload,
    PatientReportPayload,
    DoctorQueryPayload
)
from engine.ingestion import IngestionRouter

def test_normal_telemetry_passes_through():
    router = IngestionRouter()
    patient_id = "p1"
    base_time = datetime(2026, 8, 23, 12, 0, 0)
    
    # Send a series of normal telemetry payloads (HR = 80, Accel = 0.5)
    for i in range(5):
        payload = TelemetryPayload(
            patient_id=patient_id,
            timestamp=base_time + timedelta(seconds=i),
            heart_rate=80.0,
            hrv_rmssd=50.0,
            accelerometer_magnitude=0.5,
            channel_source="wearable_stream"
        )
        event = router.process_event(payload)
        
        # Verify it does not trigger LLM processing
        assert event.requires_llm_processing is False, f"Event at index {i} unexpectedly triggered LLM processing"
        assert event.channel == "wearable_anomaly"
        assert event.patient_id == patient_id
        
    # Check that normal readings are saved to local buffer
    assert len(router.local_normal_buffer) == 5

def test_anomaly_telemetry_triggers_llm():
    router = IngestionRouter()
    patient_id = "p2"
    base_time = datetime(2026, 8, 23, 12, 0, 0)
    
    # 1. Send normal readings first to fill the buffer
    for i in range(5):
        payload = TelemetryPayload(
            patient_id=patient_id,
            timestamp=base_time + timedelta(seconds=i),
            heart_rate=75.0,
            hrv_rmssd=55.0,
            accelerometer_magnitude=0.4,
            channel_source="wearable_stream"
        )
        event = router.process_event(payload)
        assert event.requires_llm_processing is False

    # 2. Send anomaly reading: HR=135, Accel=0.2 (High HR, low movement)
    # The rolling 10-second window will include the previous events, so we might need multiple or
    # a sequence that brings the average above 120.
    # If we send a couple of high HR values, the average will cross 120.
    # Let's send events that make the rolling mean HR > 120.
    # At t=6: HR=135, Accel=0.2
    payload_anomaly = TelemetryPayload(
        patient_id=patient_id,
        timestamp=base_time + timedelta(seconds=6),
        heart_rate=135.0,
        hrv_rmssd=40.0,
        accelerometer_magnitude=0.2,
        channel_source="wearable_stream"
    )
    event_anomaly = router.process_event(payload_anomaly)
    
    # Let's check what the rolling mean HR is:
    # Buffer contains: [75, 75, 75, 75, 75, 135]. Mean is 85.
    # If we want the rolling mean itself to immediately trigger, we can send a few consecutive high HRs,
    # or just send a timestamp jump so older values fall out of the 10-second window, or just have HR high enough.
    # Let's test the rolling window filtering.
    # At t=15 (which is 15 seconds after base_time), previous values (t=0 to t=4) are older than t=15 - 10s = t=5, so they fall out.
    # Thus, only t=6 and t=15 are in the window.
    # Let's send a high HR at t=15.
    payload_anomaly_2 = TelemetryPayload(
        patient_id=patient_id,
        timestamp=base_time + timedelta(seconds=15),
        heart_rate=135.0,
        hrv_rmssd=40.0,
        accelerometer_magnitude=0.2,
        channel_source="wearable_stream"
    )
    event_anomaly_2 = router.process_event(payload_anomaly_2)
    # Now buffer at t=15 has:
    # t=6 (since 6 >= 15 - 10 is false, wait! 6 < 5 is false, but 6 is >= 5, so t=6 is still in the buffer! No, 15 - 10 = 5. 6 >= 5, so yes, t=6 is in).
    # Buffer values: t=6 (HR=135, Accel=0.2) and t=15 (HR=135, Accel=0.2).
    # Rolling mean HR = 135.0, rolling mean Accel = 0.2.
    # This should trigger!
    assert event_anomaly_2.requires_llm_processing is True
    assert event_anomaly_2.channel == "wearable_anomaly"

def test_patient_and_doctor_inputs():
    router = IngestionRouter()
    base_time = datetime(2026, 8, 23, 12, 0, 0)
    
    # Test Patient Chat
    patient_payload = PatientReportPayload(
        patient_id="p3",
        timestamp=base_time,
        raw_text="I feel dizzy and lightheaded after taking my morning pill.",
        perceived_severity=8,
        channel_source="patient_chat"
    )
    event_patient = router.process_event(patient_payload)
    assert event_patient.requires_llm_processing is True
    assert event_patient.channel == "patient_chat"
    
    # Test Doctor Query
    doctor_payload = DoctorQueryPayload(
        doctor_id="d1",
        patient_id="p3",
        timestamp=base_time,
        query_text="Has the patient experienced any tachycardia spikes?",
        channel_source="doctor_portal"
    )
    event_doctor = router.process_event(doctor_payload)
    assert event_doctor.requires_llm_processing is True
    assert event_doctor.channel == "doctor_portal"

if __name__ == "__main__":
    print("Running unit tests manually...")
    test_normal_telemetry_passes_through()
    test_anomaly_telemetry_triggers_llm()
    test_patient_and_doctor_inputs()
    print("All tests passed successfully!")
