from typing import Tuple

from schemas.inputs import IngestionEvent


def _row_value(item, key):
    if item is None:
        return None
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def build_agent_context(patient_id: str, event: IngestionEvent, neo4j_client, embedder=None) -> Tuple[str, dict]:
    """
    Build a compact, structured clinical context string for the agent.
    The function returns a prompt-friendly XML/Markdown fragment plus metadata.
    """
    event_payload = event.payload
    channel = event.channel

    context = {
        "patient_id": patient_id,
        "channel": channel,
        "event_id": event.event_id,
        "event_timestamp": event.timestamp.isoformat(),
    }

    if channel == "wearable_anomaly":
        protocol_query = """
        MATCH (p:Patient {patient_id: $patient_id})
        OPTIONAL MATCH (p)-[:HAS_ANOMALY]->(a:AnomalyEvent)
        OPTIONAL MATCH (a)-[:VIOLATES]->(tp:ThresholdProtocol)
        OPTIONAL MATCH (p)-[:HAS_CONDITION]->(c:Condition)
        OPTIONAL MATCH (p)-[:HAS_PRESCRIPTION]->(rx:Prescription)
        RETURN collect(DISTINCT {
            event_id: a.event_id,
            heart_rate: a.heart_rate,
            baseline_heart_rate: a.baseline_heart_rate,
            protocol_name: tp.name,
            conditions: collect(DISTINCT c.name),
            prescriptions: collect(DISTINCT rx.name)
        }) AS anomaly_data
        """
        rows = neo4j_client.execute_query(protocol_query, {"patient_id": patient_id}) if neo4j_client else []
        recent = []
        if rows:
            row_payload = _row_value(rows[0], "anomaly_data") or []
            if isinstance(row_payload, list):
                recent = row_payload
            else:
                recent = [row_payload]

        context["recent_telemetry_anomalies"] = recent

        active_conditions = sorted({
            cond for item in recent for cond in (item.get("conditions") or []) if cond
        })
        active_protocols = sorted({
            item.get("protocol_name") for item in recent if item.get("protocol_name")
        })
        active_prescriptions = sorted({
            med for item in recent for med in (item.get("prescriptions") or []) if med
        })

        xml = f"""
<patient_profile>
  <patient_id>{patient_id}</patient_id>
  <event_id>{event.event_id}</event_id>
  <channel>{channel}</channel>
  <active_protocols>
    {''.join(f'<protocol name="{protocol}" />' for protocol in active_protocols) if active_protocols else '<protocol name="tachycardia_threshold" />'}
  </active_protocols>
  <active_conditions>
    {''.join(f'<condition>{cond}</condition>' for cond in active_conditions)}
  </active_conditions>
  <active_prescriptions>
    {''.join(f'<prescription>{med}</prescription>' for med in active_prescriptions)}
  </active_prescriptions>
  <recent_telemetry_anomalies>
    {''.join(f'<anomaly event_id="{item.get("event_id", "")}" heart_rate="{item.get("heart_rate", "")}" baseline_heart_rate="{item.get("baseline_heart_rate", "")}" protocol="{item.get("protocol_name", "tachycardia_threshold")}" />' for item in recent)}
  </recent_telemetry_anomalies>
</patient_profile>
""".strip()

    else:
        query_text = event_payload.query_text if channel == "doctor_portal" else event_payload.raw_text
        if embedder is not None:
            try:
                embedder.embed(query_text)
            except Exception:
                pass

        patient_context_query = """
        MATCH (p:Patient {patient_id: $patient_id})
        OPTIONAL MATCH (p)-[:HAS_TURN]->(t:Turn)
        OPTIONAL MATCH (t)-[:REFERENCES]->(rx:Prescription)
        OPTIONAL MATCH (p)-[:HAS_CONDITION]->(c:Condition)
        OPTIONAL MATCH (p)-[:HAS_PRESCRIPTION]->(pr:Prescription)
        RETURN collect(DISTINCT t.text) AS recent_turns,
               collect(DISTINCT rx.name) AS referenced_prescriptions,
               collect(DISTINCT c.name) AS conditions,
               collect(DISTINCT pr.name) AS prescriptions
        """
        rows = neo4j_client.execute_query(patient_context_query, {"patient_id": patient_id}) if neo4j_client else []
        row = rows[0] if rows else None
        recent_turns = _row_value(row, "recent_turns") or []
        referenced_prescriptions = _row_value(row, "referenced_prescriptions") or []
        conditions = _row_value(row, "conditions") or []
        prescriptions = _row_value(row, "prescriptions") or []

        context["recent_turns"] = recent_turns
        context["referenced_prescriptions"] = referenced_prescriptions
        context["conditions"] = conditions
        context["prescriptions"] = prescriptions

        xml = f"""
<patient_profile>
  <patient_id>{patient_id}</patient_id>
  <event_id>{event.event_id}</event_id>
  <channel>{channel}</channel>
  <query_or_report>
    {query_text}
  </query_or_report>
  <recent_turns>
    {''.join(f'<turn>{turn}</turn>' for turn in recent_turns[:5])}
  </recent_turns>
  <referenced_prescriptions>
    {''.join(f'<prescription>{med}</prescription>' for med in referenced_prescriptions)}
  </referenced_prescriptions>
  <conditions>
    {''.join(f'<condition>{cond}</condition>' for cond in conditions)}
  </conditions>
  <prescriptions>
    {''.join(f'<prescription>{med}</prescription>' for med in prescriptions)}
  </prescriptions>
</patient_profile>
""".strip()

    context["prompt_context"] = xml
    return xml, context
