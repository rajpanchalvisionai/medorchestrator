from datetime import datetime
from typing import Literal, Union, List, Optional
from pydantic import BaseModel, Field

class NormalizedEntity(BaseModel):
    raw_text: str
    canonical_name: str
    code_type: Literal["RxNorm", "SNOMED", "LOINC"]
    code_id: str
    confidence: float

class TelemetryPayload(BaseModel):
    patient_id: str
    timestamp: datetime
    heart_rate: float = Field(..., description="Heart Rate in BPM")
    hrv_rmssd: float = Field(..., description="HRV RMSSD in ms")
    accelerometer_magnitude: float = Field(..., description="Accelerometer magnitude in g-force")
    channel_source: Literal["wearable_stream"] = "wearable_stream"

class PatientReportPayload(BaseModel):
    patient_id: str
    timestamp: datetime
    raw_text: str
    perceived_severity: int = Field(5, ge=1, le=10, description="Perceived severity on a 1-10 scale")
    channel_source: Literal["patient_chat"] = "patient_chat"
    normalized_entities: List[NormalizedEntity] = Field(default_factory=list)

class DoctorQueryPayload(BaseModel):
    doctor_id: str
    patient_id: str
    timestamp: datetime
    query_text: str
    channel_source: Literal["doctor_portal"] = "doctor_portal"
    normalized_entities: List[NormalizedEntity] = Field(default_factory=list)

class IngestionEvent(BaseModel):
    event_id: str
    patient_id: str
    channel: Literal["wearable_anomaly", "patient_chat", "doctor_portal"]
    timestamp: datetime
    payload: Union[TelemetryPayload, PatientReportPayload, DoctorQueryPayload]
    requires_llm_processing: bool
    entropy_score: Optional[float] = Field(default=None, description="Density gating entropy score H(W_t)")
    baseline_diff: Optional[float] = Field(default=None, description="Deviation from nominal baseline HR")
