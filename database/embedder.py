import os
from collections import OrderedDict
from typing import List

try:
    from google import genai
    from google.genai import Client
except Exception:  # pragma: no cover
    genai = None
    Client = None


class GeminiEmbedder:
    """gemini-embedding-2 adapter with an LRU-like cache for identical text."""

    def __init__(self, api_key: str | None = None, cache_size: int = 256):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.cache: OrderedDict[str, List[float]] = OrderedDict()
        self.cache_size = cache_size
        self.client = None
        if genai is not None and Client is not None and self.api_key:
            self.client = Client(api_key=self.api_key)

    def embed(self, text: str) -> List[float]:
        if text in self.cache:
            value = self.cache.pop(text)
            self.cache[text] = value
            return list(value)

        if self.client is None:
            # Graceful fallback: return a deterministic zero vector for tests/offline runs.
            vector = [0.0] * 768
            self.cache[text] = vector
            self.cache.move_to_end(text)
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
            return list(vector)

        try:
            response = self.client.models.embed_content(
                model="gemini-embedding-2",
                contents=text,
            )
            embeddings = getattr(response, 'embeddings', None)
            if embeddings and len(embeddings) > 0:
                embedding = embeddings[0]
                values = getattr(embedding, 'values', None)
                if values is None and isinstance(embedding, dict):
                    values = embedding.get('values')
                if values is None:
                    values = list(embedding)
                vector = list(values)
            else:
                vector = [0.0] * 768
        except Exception:
            vector = [0.0] * 768

        if len(vector) != 768:
            vector = vector[:768] + [0.0] * max(0, 768 - len(vector))

        self.cache[text] = vector
        self.cache.move_to_end(text)
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return list(vector)
