import os
import pytest

pytest.importorskip("neo4j")

from database.bolt_client import Neo4jBoltClient
from database.embedder import GeminiEmbedder
from memory.graph_writer import seed_patient_profile, write_anomaly_event, write_chat_turn
from memory.graph_retriever import hybrid_vector_graph_retrieval, check_patient_discrepancy


@pytest.mark.skipif(
    not all(os.getenv(k) for k in ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"]),
    reason="Neo4j environment variables are not configured",
)
def test_stage3_graph_memory_smoke():
    client = Neo4jBoltClient.from_env()
    try:
        client.init_db_schema()
        assert client is not None
    finally:
        client.close()


@pytest.mark.skipif(
    not all(os.getenv(k) for k in ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "GEMINI_API_KEY"]),
    reason="Neo4j and Gemini environment variables are not configured",
)
def test_stage3_graph_memory_embedding_and_seed_smoke(tmp_path):
    client = Neo4jBoltClient.from_env()
    embedder = GeminiEmbedder(api_key=os.getenv("GEMINI_API_KEY"))

    try:
        client.init_db_schema()

        patient_id = "patient_neo4j_smoke"
        user_info_path = tmp_path / "user_info.csv"
        user_info_path.write_text("patient_id,name,medication\npatient_neo4j_smoke,Example Patient,Metoprolol\n", encoding="utf-8")

        seed_patient_profile(client, str(user_info_path), patient_id)

        event = {
            "patient_id": patient_id,
            "heart_rate": 133,
            "baseline_heart_rate": 75,
            "channel": "wearable_anomaly",
            "clinical_context": "tachycardia event violates threshold protocol",
        }
        anomaly = write_anomaly_event(client, event)
        assert anomaly is not None

        turn = write_chat_turn(client, embedder, {
            "patient_id": patient_id,
            "text": "I took Metoprolol and felt dizzy.",
            "author": "patient",
            "channel": "patient_chat",
        })
        assert turn is not None

        results = hybrid_vector_graph_retrieval(client, embedder, "Metoprolol and dizziness", patient_id)
        assert isinstance(results, list)

        discrepancy = check_patient_discrepancy(client, patient_id, event)
        assert discrepancy in (True, False)
    finally:
        client.close()
