"""Clinical reasoning agent package for Stage 4."""

from .clinical_agent import process_ingestion_event
from .context_builder import build_agent_context

__all__ = ["build_agent_context", "process_ingestion_event"]
