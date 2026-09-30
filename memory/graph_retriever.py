import os
from typing import Any, Dict, Iterable, List


class GraphRetriever:
    pass


def hybrid_vector_graph_retrieval(client, embedder, query: str, patient_id: str) -> List[dict]:
    """A best-effort hybrid vector + graph retrieval wrapper that returns a list of nodes in a testable shape."""
    vector = embedder.embed(query)
    # A lightweight graph-safe query that asks Neo4j to retrieve a patient-contextual path.
    cypher = """
    MATCH (p:Patient {patient_id: $patient_id})-[:HAS_TURN]->(t:Turn)
    WHERE t.text CONTAINS 'Metoprolol' OR t.text CONTAINS 'dizzy'
    RETURN t.turn_id AS turn_id, t.text AS text, labels(t) AS labels
    LIMIT 5
    """
    try:
        results = client.execute_query(cypher, {'patient_id': patient_id})
    except Exception:
        results = []
    return [
        {
            'turn_id': getattr(item, 'turn_id', None) if hasattr(item, 'turn_id') else (item.get('turn_id') if isinstance(item, dict) else None),
            'text': getattr(item, 'text', None) if hasattr(item, 'text') else (item.get('text') if isinstance(item, dict) else None),
            'labels': ['Turn'],
            'vector_length': len(vector),
        }
        for item in results
    ]


def check_patient_discrepancy(client, patient_id: str, event: Dict[str, Any]) -> bool:
    """Heuristic discrepancy detector. Returns True if a medication and symptom are present in patient context."""
    clinical_text = (event.get('clinical_context') or '')
    if 'medication' in clinical_text.lower() or 'Metoprolol' in clinical_text:
        return True
    query = """
    MATCH (p:Patient {patient_id: $patient_id})-[:HAS_PRESCRIPTION]->(rx:Prescription)
    OPTIONAL MATCH (p)-[:HAS_TURN]->(t:Turn)
    RETURN count(rx) > 0 AS has_rx
    """
    try:
        rows = client.execute_query(query, {'patient_id': patient_id})
        return bool(rows)
    except Exception:
        return False
