import asyncio
import logging
from typing import Iterable

from agent.clinical_agent import process_ingestion_event
from schemas.inputs import IngestionEvent

logger = logging.getLogger("AgentPipelineRunner")


async def run_agent_pipeline(events: Iterable[IngestionEvent], neo4j_client=None, embedder=None):
    decisions = []
    for event in events:
        if not event.requires_llm_processing:
            continue
        decision = process_ingestion_event(event, neo4j_client=neo4j_client, embedder=embedder)
        decisions.append(decision)
        logger.info("Agent decision produced for %s/%s: %s", event.patient_id, event.channel, decision.recommended_action)
    return decisions
