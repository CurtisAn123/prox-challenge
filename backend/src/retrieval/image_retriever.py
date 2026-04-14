"""
src/retrieval/image_retriever.py
──────────────────────────────────
Retrieves images relevant to a user query and converts them to base64
so the API can return them inline.

Flow:
  1. VectorStore.search_images() returns caption-ranked results.
  2. ImageRetriever loads the actual PNG bytes from disk.
  3. Returns a list of ImageResult objects (base64 payload + metadata).
"""

import base64
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from src.ingestion.vector_store import VectorStore
from config.settings import get_settings


@dataclass
class ImageResult:
    """A retrieved image ready to be included in an API response."""
    file_path: str
    page: int
    source: str
    caption: str
    score: float
    base64_data: str        # PNG bytes, base64-encoded
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ImageRetriever:
    """
    Thin wrapper around VectorStore.search_images() that also loads
    the raw image bytes from disk.
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store
        self.settings = get_settings()

    def retrieve(self, query: str, top_k: int | None = None) -> list[ImageResult]:
        """
        Search for images relevant to *query* and return them with their
        base64-encoded PNG data for inline API delivery.

        The *score* field is cosine similarity (0→1). Only images above
        0.30 similarity are returned to filter out noise.
        """
        raw_results = self.vector_store.search_images(
            query, top_k=top_k or self.settings.top_k_images
        )

        results: list[ImageResult] = []
        for r in raw_results:
            # Similarity threshold: skip clearly irrelevant images
            if r["score"] < 0.30:
                continue

            file_path = r.get("file_path", "")
            if not file_path or not Path(file_path).exists():
                continue

            # Load and base64-encode the image
            try:
                with open(file_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
            except OSError:
                continue

            results.append(
                ImageResult(
                    file_path=file_path,
                    page=r.get("page", 0),
                    source=r.get("source", "unknown"),
                    caption=r.get("caption", ""),
                    score=r.get("score", 0.0),
                    base64_data=b64,
                    width=r.get("width", 0),
                    height=r.get("height", 0),
                )
            )

        return results
