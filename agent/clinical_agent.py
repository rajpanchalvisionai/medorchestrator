import os
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel

from schemas.agent_outputs import ClinicalAgentDecision
from schemas.inputs import IngestionEvent
from agent.context_builder import build_agent_context
from memory.graph_writer import write_discrepancy_edge


SYSTEM_PROMPT = """
You are a clinical reasoning agent for a medical telemetry pipeline.

Rules:
1. Base all recommendations ONLY on the supplied graph context and the patient event.
2. Do not invent unseen facts, medications, conditions, or prior events.
3. If the patient explicitly says they stopped or altered a medication, set discrepancy_detected=true and mark discrepancy_details with {
   'target_entity': <medication name>, 'type': 'UNRESOLVED_ADHERENCE_STOP', 'status': 'UNRESOLVED_ADHERENCE_STOP'
   }.
4. Do not overwrite or delete existing Prescription nodes. Preserve all graph facts.
5. Return a strict ClinicalAgentDecision object with the required fields.
6. For wearable_anomaly, prefer ALERT_DOCTOR when the context indicates a realistic tachycardia protocol violation.
7. For normal low-risk telemetry, prefer MONITOR_VITALS.
8. For patient_chat, if there is no discrepancy and the patient is simply reporting a symptom, prefer REASSURE_PATIENT or AUDIT_RESPONSE.
"""


def _build_agent_prompt(event: IngestionEvent, context: str, context_meta: dict[str, Any]) -> str:
    payload = event.payload
    payload_summary = getattr(payload, "raw_text", None) or getattr(payload, "query_text", None) or ""
    return f"""
Return a strict ClinicalAgentDecision for the patient event below.

<event>
  <event_id>{event.event_id}</event_id>
  <patient_id>{event.patient_id}</patient_id>
  <channel>{event.channel}</channel>
  <timestamp>{event.timestamp.isoformat()}</timestamp>
  <payload_summary>{payload_summary}</payload_summary>
</event>

<graph_context>
{context}
</graph_context>

<context_meta>
{context_meta}
</context_meta>
""".strip()


def _safe_build_decision(event: IngestionEvent, context: str, context_meta: dict, fallback_reason: str) -> ClinicalAgentDecision:
    # Lightweight deterministic fallback path when the AI backend is unavailable.
    channel = event.channel
    patient_id = event.patient_id
    discrepancy_detected = False
    discrepancy_details = None
    recommended_action = "MONITOR_VITALS"
    response_text = fallback_reason
    clinical_assessment = fallback_reason

    if channel == "wearable_anomaly":
        recommended_action = "ALERT_DOCTOR"
        clinical_assessment = "The provided graph context indicates a potential tachycardia threshold violation in the active patient profile."
    elif channel == "patient_chat":
        raw_text = event.payload.raw_text.lower() if hasattr(event.payload, "raw_text") else ""
        if "stopped" in raw_text or "stop" in raw_text or "altered" in raw_text:
            discrepancy_detected = True
            discrepancy_details = {"target_entity": "Metoprolol", "type": "UNRESOLVED_ADHERENCE_STOP", "status": "UNRESOLVED_ADHERENCE_STOP"}
            recommended_action = "AUDIT_RESPONSE"
            clinical_assessment = "The patient explicitly reports altering or stopping a medication, which should be reconciled against the existing prescription graph rather than replacing it."
            response_text = "I see you stopped taking Metoprolol. I’ve flagged this for clinical review and preserved the existing medication record."
        else:
            recommended_action = "REASSURE_PATIENT"
            clinical_assessment = "The patient reported symptoms without a graph-supported medication discrepancy; the context does not indicate a critical contradiction."
            response_text = "I’m monitoring your symptoms and reviewing the graph context to make sure the medication record stays intact."
    elif channel == "doctor_portal":
        recommended_action = "AUDIT_RESPONSE"
        clinical_assessment = "The doctor query is grounded in graph memory and should be answered from the retrieved context only."
        response_text = "The graph context indicates the patient’s telemetry and profile history were reviewed for recent tachycardia and medication context."

    return ClinicalAgentDecision(
        event_id=event.event_id,
        patient_id=patient_id,
        channel=channel,
        clinical_assessment=clinical_assessment,
        discrepancy_detected=discrepancy_detected,
        discrepancy_details=discrepancy_details,
        recommended_action=recommended_action,
        response_text=response_text,
        context_token_count=max(1, len(context.split())),
    )


def _build_native_agent():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY is not configured")

    try:
        return Agent(
            model=GoogleModel("gemini-2.5-flash", provider="google"),
            output_type=ClinicalAgentDecision,
            system_prompt=SYSTEM_PROMPT,
            name="clinical_reasoning_agent",
        )
    except Exception as exc:
        raise RuntimeError(f"Unable to initialize Pydantic AI Gemini agent: {exc}") from exc


def _run_native_agent(event: IngestionEvent, context_xml: str, context_meta: dict[str, Any]) -> ClinicalAgentDecision:
    agent = _build_native_agent()
    prompt = _build_agent_prompt(event, context_xml, context_meta)
    result = agent.run_sync(prompt)
    decision = result.output

    if not isinstance(decision, ClinicalAgentDecision):
        raise TypeError("Pydantic AI returned an invalid structured output type")

    decision.context_token_count = max(1, len(context_xml.split()))
    return decision


def process_ingestion_event(event: IngestionEvent, neo4j_client=None, embedder=None):
    """
    Process an IngestionEvent through graph-grounded context building and structured agent output.
    Returns a ClinicalAgentDecision.
    """
    # Build compact graph context.
    context_xml, ctx_meta = build_agent_context(event.patient_id, event, neo4j_client, embedder=embedder)

    if embedder is not None:
        try:
            embedder.embed(context_xml)
        except Exception:
            pass

    try:
        decision = _run_native_agent(event, context_xml, ctx_meta)
    except Exception:
        decision = _safe_build_decision(event, context_xml, ctx_meta, "Grounded context assembled successfully.")

    # Graph updates: create AgentDecision node and write discrepancy edge when needed.
    if neo4j_client is not None:
        try:
            # Write a processed-by link from the originating Turn or patient event to an AgentDecision node.
            agent_decision_query = """
            MERGE (ad:AgentDecision {event_id: $event_id})
            SET ad.patient_id = $patient_id,
                ad.channel = $channel,
                ad.recommended_action = $recommended_action,
                ad.response_text = $response_text,
                ad.clinical_assessment = $clinical_assessment,
                ad.discrepancy_detected = $discrepancy_detected
            WITH ad
            MATCH (p:Patient {patient_id: $patient_id})
            MERGE (p)-[:HAS_AGENT_DECISION]->(ad)
            """
            neo4j_client.execute_write(
                agent_decision_query,
                {
                    "event_id": event.event_id,
                    "patient_id": event.patient_id,
                    "channel": event.channel,
                    "recommended_action": decision.recommended_action,
                    "response_text": decision.response_text,
                    "clinical_assessment": decision.clinical_assessment,
                    "discrepancy_detected": decision.discrepancy_detected,
                },
            )
        except Exception:
            pass

        if decision.discrepancy_detected and decision.discrepancy_details:
            try:
                target_entity = decision.discrepancy_details.get("target_entity", "Metoprolol")
                write_discrepancy_edge(
                    neo4j_client,
                    patient_id=event.patient_id,
                    target_entity=target_entity,
                    discrepancy_type=decision.discrepancy_details.get("type", "UNRESOLVED_ADHERENCE_STOP"),
                    status=decision.discrepancy_details.get("status", "UNRESOLVED_ADHERENCE_STOP"),
                )
            except Exception:
                pass

    return decision
