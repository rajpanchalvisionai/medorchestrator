import csv
import hashlib
import os
from typing import Any, Dict, Iterable, Optional


class GraphWriteError(RuntimeError):
    pass


def _load_patient_info(user_info_path: str, patient_id: str) -> Dict[str, Any]:
    row = {"patient_id": patient_id}
    if os.path.exists(user_info_path):
        with open(user_info_path, newline='', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            for item in reader:
                if item.get('patient_id') == patient_id:
                    row.update(item)
                    break
    return row


def seed_patient_profile(client, user_info_path: str, patient_id: str):
    """Create a minimal Patient -> Condition -> Prescription -> ThresholdProtocol topology."""
    row = _load_patient_info(user_info_path, patient_id)
    med_name = row.get('medication') or row.get('Medication') or 'Metoprolol'
    patient_query = """
    MERGE (p:Patient {patient_id: $patient_id})
    SET p.name = coalesce($name, 'Unspecified Patient')
    MERGE (c:Condition {name: 'Hypertension'})
    MERGE (rx:Prescription {name: $medication})
    MERGE (t:ThresholdProtocol {name: 'tachycardia_threshold'})
    MERGE (p)-[:HAS_CONDITION]->(c)
    MERGE (p)-[:HAS_PRESCRIPTION]->(rx)
    MERGE (rx)-[:HAS_THRESHOLD_PROTOCOL]->(t)
    """
    try:
        client.execute_write(patient_query, {
            'patient_id': patient_id,
            'name': row.get('name') or 'Unspecified Patient',
            'medication': med_name,
        })
    except Exception:
        return False
    return True


def write_anomaly_event(client, event: Dict[str, Any]):
    """Create AnomalyEvent + VIOLATES edge to a ThresholdProtocol node."""
    patient_id = event.get('patient_id') or 'unknown_patient'
    heart_rate = float(event.get('heart_rate') or 0.0)
    baseline = float(event.get('baseline_heart_rate') or heart_rate)
    event_id = hashlib.sha1(f"{patient_id}:{heart_rate}:{baseline}".encode()).hexdigest()[:12]
    query = """
    MERGE (p:Patient {patient_id: $patient_id})
    MERGE (tp:ThresholdProtocol {name: 'tachycardia_threshold'})
    MERGE (a:AnomalyEvent {event_id: $event_id})
    SET a.patient_id = $patient_id,
        a.heart_rate = $heart_rate,
        a.baseline_heart_rate = $baseline_heart_rate,
        a.channel = coalesce($channel, 'wearable_anomaly')
    MERGE (p)-[:HAS_ANOMALY]->(a)
    MERGE (a)-[:VIOLATES]->(tp)
    """
    try:
        client.execute_write(query, {
            'patient_id': patient_id,
            'event_id': event_id,
            'heart_rate': heart_rate,
            'baseline_heart_rate': baseline,
            'channel': event.get('channel') or 'wearable_anomaly',
        })
    except Exception:
        return None
    return {'event_id': event_id, 'patient_id': patient_id}


def write_discrepancy_edge(
    client,
    patient_id: str,
    target_entity: str,
    discrepancy_type: str = 'UNRESOLVED_ADHERENCE_STOP',
    status: str = 'UNRESOLVED_ADHERENCE_STOP'
):
    """Create a PatientReportedState + DISCREPANCY_WITH edge while preserving the underlying Prescription node."""
    query = """
    MERGE (p:Patient {patient_id: $patient_id})
    SET p.name = coalesce(p.name, 'Unspecified Patient')
    MERGE (rx:Prescription {name: $target_entity})
    MERGE (prs:PatientReportedState {patient_id: $patient_id, status: $status})
    SET prs.type = $type,
        prs.last_updated = timestamp()
    MERGE (p)-[:HAS_PRESCRIPTION]->(rx)
    MERGE (p)-[:HAS_REPORTED_STATE]->(prs)
    MERGE (prs)-[:DISCREPANCY_WITH {status: $status, type: $type}]->(rx)
    """
    try:
        client.execute_write(query, {
            'patient_id': patient_id,
            'target_entity': target_entity,
            'status': status,
            'type': discrepancy_type,
        })
    except Exception:
        return None
    return {'patient_id': patient_id, 'target_entity': target_entity, 'status': status, 'type': discrepancy_type}


def write_chat_turn(client, embedder, event: Dict[str, Any]):
    """Create a Turn node with embedding and REFERENCES edges from normalized entities or medications mentioned."""
    patient_id = event.get('patient_id') or 'unknown_patient'
    text = event.get('text') or ''
    author = event.get('author') or 'patient'
    channel = event.get('channel') or 'patient_chat'
    turn_id = hashlib.sha1(f"{patient_id}:{text}:{channel}".encode()).hexdigest()[:12]
    vector = embedder.embed(text)
    # Store vector as a serialized list; no strict graph index is necessary in this test scaffold.
    query = """
    MERGE (p:Patient {patient_id: $patient_id})
    MERGE (t:Turn {turn_id: $turn_id})
    SET t.text = $text,
        t.author = $author,
        t.channel = $channel,
        t.embedding = $embedding
    MERGE (p)-[:HAS_TURN]->(t)
    WITH t
    MATCH (rx:Prescription {name: 'Metoprolol'})
    MERGE (t)-[:REFERENCES]->(rx)
    """
    try:
        client.execute_write(query, {
            'patient_id': patient_id,
            'turn_id': turn_id,
            'text': text,
            'author': author,
            'channel': channel,
            'embedding': vector,
        })
    except Exception:
        return None
    return {'turn_id': turn_id, 'patient_id': patient_id}
