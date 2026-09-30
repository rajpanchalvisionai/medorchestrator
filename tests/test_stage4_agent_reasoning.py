import os

import pytest

from agent.clinical_agent import process_ingestion_event
from database.bolt_client import Neo4jBoltClient
from database.embedder import GeminiEmbedder
from engine.mmash_loader import MMASHDatasetLoader
from schemas.inputs import IngestionEvent, PatientReportPayload

USER_1_DIR = os.path.join(os.path.dirname(__file__), "..", "DataPaper", "user_1")


@pytest.mark.skipif(
    not all(os.getenv(k) for k in ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "GEMINI_API_KEY"]),
    reason="Neo4j and Gemini env vars must be configured",
)
def test_stage4_anomaly_resolution_uses_graph_context():
    client = Neo4jBoltClient.from_env()
    embedder = GeminiEmbedder(api_key=os.getenv("GEMINI_API_KEY"))
    try:
        client.init_db_schema()
        loader = MMASHDatasetLoader()
        patient_id = "user_1"
        telemetry = list(loader.stream_telemetry(USER_1_DIR, patient_id=patient_id, limit=30, start_row=0))
        anomaly_event = None
        for item in telemetry:
            from engine.ingestion import IngestionRouter
            router = IngestionRouter()
            event = router.process_event(item)
            if event.requires_llm_processing:
                anomaly_event = event
                break
        assert anomaly_event is not None
        decision = process_ingestion_event(anomaly_event, neo4j_client=client, embedder=embedder)
        assert decision.recommended_action in {"ALERT_DOCTOR", "MONITOR_VITALS"}
        assert decision.context_token_count < 500
        assert decision.clinical_assessment
    finally:
        client.close()


@pytest.mark.skipif(
    not all(os.getenv(k) for k in ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "GEMINI_API_KEY"]),
    reason="Neo4j and Gemini env vars must be configured",
)
def test_stage4_epistemic_reconciliation_creates_discrepancy_edge():
    client = Neo4jBoltClient.from_env()
    embedder = GeminiEmbedder(api_key=os.getenv("GEMINI_API_KEY"))
    try:
        client.init_db_schema()
        patient_id = "user_1"
        event = IngestionEvent(
            event_id="stage4-epistemic-1",
            patient_id=patient_id,
            channel="patient_chat",
            timestamp=__import__("datetime").datetime.now(),
            payload=PatientReportPayload(
                patient_id=patient_id,
                timestamp=__import__("datetime").datetime.now(),
                raw_text="I stopped taking my Metoprolol because it made me dizzy.",
                perceived_severity=7,
            ),
            requires_llm_processing=True,
        )
        decision = process_ingestion_event(event, neo4j_client=client, embedder=embedder)
        assert decision.discrepancy_detected is True
        assert decision.discrepancy_details is not None
        assert decision.discrepancy_details["target_entity"] == "Metoprolol"

        rows = client.execute_query(
            """
            MATCH (p:Patient {patient_id: $patient_id})-[:HAS_PRESCRIPTION]->(rx:Prescription {name: 'Metoprolol'})
            OPTIONAL MATCH (prs:PatientReportedState {patient_id: $patient_id})-[:DISCREPANCY_WITH]->(rx)
            RETURN count(prs) AS discrepancy_count, rx.name AS medication
            """,
            {"patient_id": patient_id},
        )
        assert len(rows) >= 1
    finally:
        client.close()


@pytest.mark.skipif(
    not all(os.getenv(k) for k in ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "GEMINI_API_KEY"]),
    reason="Neo4j and Gemini env vars must be configured",
)
def test_stage4_doctor_query_accuracy_from_graph_memory():
    client = Neo4jBoltClient.from_env()
    embedder = GeminiEmbedder(api_key=os.getenv("GEMINI_API_KEY"))
    try:
        client.init_db_schema()
        loader = MMASHDatasetLoader()
        patient_id = "user_1"
        queries = loader.generate_doctor_queries(USER_1_DIR, patient_id=patient_id)
        doctor_query = queries[0]
        event = IngestionEvent(
            event_id="stage4-doctor-1",
            patient_id=patient_id,
            channel="doctor_portal",
            timestamp=doctor_query.timestamp,
            payload=doctor_query,
            requires_llm_processing=True,
        )
        decision = process_ingestion_event(event, neo4j_client=client, embedder=embedder)
        assert decision.response_text
        assert "tachycardia" in decision.response_text.lower() or "graph" in decision.response_text.lower()
    finally:
        client.close()


@pytest.mark.skipif(
    not all(os.getenv(k) for k in ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "GEMINI_API_KEY"]),
    reason="Neo4j and Gemini env vars must be configured",
)
def test_stage4_context_token_compression_under_500():
    client = Neo4jBoltClient.from_env()
    embedder = GeminiEmbedder(api_key=os.getenv("GEMINI_API_KEY"))
    try:
        client.init_db_schema()
        loader = MMASHDatasetLoader()
        patient_id = "user_1"
        telemetry = list(loader.stream_telemetry(USER_1_DIR, patient_id=patient_id, limit=20, start_row=0))
        from engine.ingestion import IngestionRouter
        router = IngestionRouter()
        event = router.process_event(telemetry[0])
        decision = process_ingestion_event(event, neo4j_client=client, embedder=embedder)
        assert decision.context_token_count < 500
    finally:
        client.close()
