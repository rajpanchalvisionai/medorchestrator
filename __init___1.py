"""Graph database and embedding adapters for the HealthOrchestrator Stage 3 layer."""

import os
from pathlib import Path


def load_env_file():
    """Load a workspace .env file into os.environ without requiring python-dotenv."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"\'')
        os.environ.setdefault(key, value)


load_env_file()
