from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ClinicalAgentDecision(BaseModel):
    event_id: str
    patient_id: str
    channel: Literal["wearable_anomaly", "patient_chat", "doctor_portal"]
    clinical_assessment: str
    discrepancy_detected: bool = False
    discrepancy_details: Optional[dict] = Field(default=None)
    recommended_action: Literal["ALERT_DOCTOR", "MONITOR_VITALS", "REASSURE_PATIENT", "AUDIT_RESPONSE"]
    response_text: str
    context_token_count: int = Field(default=0, ge=0)

    model_config = ConfigDict(extra="forbid")
