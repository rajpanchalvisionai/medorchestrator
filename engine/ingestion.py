import uuid
from datetime import datetime
from typing import Dict, List, Optional, Union

from schemas.inputs import (
    TelemetryPayload,
    PatientReportPayload,
    DoctorQueryPayload,
    IngestionEvent
)
from engine.signal_processor import PolarsTelemetryBuffer, GatingResult
from engine.normalizer import ClinicalTextNormalizer

class IngestionRouter:
    """
    High-Throughput Multi-Channel Ingestion Router.
    Integrates Polars time-series windowing with H(W_t) density gating for wearable telemetry,
    and ScispaCy/spaCy canonical medical entity normalization (RxNorm/SNOMED/LOINC) for text channels.
    """
    def __init__(
        self,
        window_seconds: float = 10.0,
        nominal_baseline_hr: float = 75.0,
        accel_threshold: float = 1.0
    ):
        self.window_seconds = window_seconds
        self.nominal_baseline_hr = nominal_baseline_hr
        self.accel_threshold = accel_threshold

        # Maps patient_id -> PolarsTelemetryBuffer
        self.telemetry_buffers: Dict[str, PolarsTelemetryBuffer] = {}
        # Quiet local buffer for normal telemetry readings
        self.local_normal_buffer: List[TelemetryPayload] = []
        # Canonical clinical text entity normalizer
        self.normalizer = ClinicalTextNormalizer()

    def get_or_create_buffer(self, patient_id: str) -> PolarsTelemetryBuffer:
        if patient_id not in self.telemetry_buffers:
            self.telemetry_buffers[patient_id] = PolarsTelemetryBuffer(
                window_seconds=self.window_seconds,
                nominal_baseline_hr=self.nominal_baseline_hr,
                accel_threshold=self.accel_threshold
            )
        return self.telemetry_buffers[patient_id]

    def process_event(
        self,
        payload: Union[TelemetryPayload, PatientReportPayload, DoctorQueryPayload]
    ) -> IngestionEvent:
        event_id = str(uuid.uuid4())

        # CHANNEL 1: Continuous Wearable Telemetry
        if isinstance(payload, TelemetryPayload):
            buffer = self.get_or_create_buffer(payload.patient_id)
            gating: GatingResult = buffer.append(payload)

            # If H(W_t) == 1.0 (sustained tachycardia during rest), flag anomaly
            if gating.entropy_score == 1.0:
                channel = "wearable_anomaly"
                requires_llm = True
            else:
                channel = "wearable_anomaly"
                requires_llm = False
                # Log quietly to fast local buffer without waking downstream LLM
                self.local_normal_buffer.append(payload)

            return IngestionEvent(
                event_id=event_id,
                patient_id=payload.patient_id,
                channel=channel,
                timestamp=payload.timestamp,
                payload=payload,
                requires_llm_processing=requires_llm,
                entropy_score=gating.entropy_score,
                baseline_diff=gating.baseline_diff
            )

        # CHANNEL 2: Patient Self-Reported Ailment (Patient Chat)
        elif isinstance(payload, PatientReportPayload):
            # Intercept free-text and resolve canonical RxNorm / SNOMED entities
            normalized_entities = self.normalizer.normalize(payload.raw_text)
            payload.normalized_entities = normalized_entities

            return IngestionEvent(
                event_id=event_id,
                patient_id=payload.patient_id,
                channel="patient_chat",
                timestamp=payload.timestamp,
                payload=payload,
                requires_llm_processing=True,
                entropy_score=None,
                baseline_diff=None
            )

        # CHANNEL 3: Doctor Query / Clinical Audit (Doctor Portal)
        elif isinstance(payload, DoctorQueryPayload):
            # Intercept clinical query text and resolve canonical medical entities
            normalized_entities = self.normalizer.normalize(payload.query_text)
            payload.normalized_entities = normalized_entities

            return IngestionEvent(
                event_id=event_id,
                patient_id=payload.patient_id,
                channel="doctor_portal",
                timestamp=payload.timestamp,
                payload=payload,
                requires_llm_processing=True,
                entropy_score=None,
                baseline_diff=None
            )

        else:
            raise ValueError(f"Unknown payload type: {type(payload)}")
