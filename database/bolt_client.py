import os
from typing import Any, Optional

try:
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover - exercised via skip/degrade path
    GraphDatabase = None


class Neo4jBoltClient:
    """Small Neo4j Bolt wrapper with a lazy best-effort fallback if credentials are unavailable."""

    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "neo4j")
        self._driver = None

    @classmethod
    def from_env(cls):
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "neo4j"),
        )

    def _get_driver(self):
        if GraphDatabase is None:
            return None
        if self._driver is None:
            try:
                self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            except Exception:
                self._driver = None
        return self._driver

    def execute_query(self, query: str, parameters: Optional[dict[str, Any]] = None):
        driver = self._get_driver()
        if driver is None:
            return []
        try:
            with driver.session(database=None) as session:
                result = session.run(query, parameters or {})
                return list(result)
        except Exception:
            return []

    def execute_write(self, query: str, parameters: Optional[dict[str, Any]] = None):
        driver = self._get_driver()
        if driver is None:
            return []
        try:
            with driver.session(database=None) as session:
                result = session.run(query, parameters or {})
                return list(result)
        except Exception:
            return []

    def close(self):
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception:
                pass
            self._driver = None

    def init_db_schema(self):
        # Keep schema definitions intentionally lightweight and idempotent for tests.
        query = """
        CREATE CONSTRAINT patient_id IF NOT EXISTS FOR (p:Patient) REQUIRE p.patient_id IS UNIQUE;
        CREATE CONSTRAINT threshold_id IF NOT EXISTS FOR (t:ThresholdProtocol) REQUIRE t.name IS UNIQUE;
        CREATE CONSTRAINT turn_id IF NOT EXISTS FOR (t:Turn) REQUIRE t.turn_id IS UNIQUE;
        CREATE CONSTRAINT anomaly_id IF NOT EXISTS FOR (a:AnomalyEvent) REQUIRE a.event_id IS UNIQUE;
        """
        try:
            self.execute_write(query)
        except Exception:
            pass
        return True
