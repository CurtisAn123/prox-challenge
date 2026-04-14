"""
config/settings.py
──────────────────
Centralised configuration loaded from the .env file.

All tunable parameters live here so they never have to be changed
inside application code. Import via `from config import get_settings`.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Product identity ─────────────────────────────────────────────────────
    # Used in system prompts — change this when pointing the agent at a
    # different product manual without touching any reasoning code.
    product_name: str = "Vulcan OmniPro 220"

    # ── Mistral ──────────────────────────────────────────────────────────────
    mistral_api_key: str = Field(..., description="Mistral API key for OCR")
    mistral_ocr_model: str = "mistral-ocr-latest"

    # ── Anthropic ───────────────────────────────────────────────────────────
    anthropic_api_key: str = Field(..., description="Anthropic API key")

    # Primary reasoning model (best accuracy for complex welding queries)
    claude_model: str = "claude-opus-4-6"

    # Fast classification model (intent routing — cheap, quick)
    claude_fast_model: str = "claude-haiku-4-5-20251001"

    # Vision model for captioning extracted PDF images
    claude_vision_model: str = "claude-opus-4-6"

    # ── Paths ────────────────────────────────────────────────────────────────
    # PDFs to ingest (relative to the repo root)
    files_dir: Path = Path("../files")

    # ChromaDB persistence directory (created automatically on first run)
    chroma_db_path: str = "./data/chroma"

    # Pickle file for the serialised NetworkX knowledge graph
    graph_path: str = "./data/knowledge_graph.pkl"

    # Directory where page images extracted from the PDF are saved
    extracted_images_dir: Path = Path("./data/images")

    # ── Embedding model ──────────────────────────────────────────────────────
    # Local sentence-transformers model — no second API key needed
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384   # must match model output dimension

    # ── Text chunking ────────────────────────────────────────────────────────
    chunk_size: int = 800    # characters per chunk
    chunk_overlap: int = 150  # overlap between consecutive chunks

    # ── Retrieval ────────────────────────────────────────────────────────────
    # Number of text chunks returned for RAG context
    top_k_text: int = 10

    # Number of image results returned for image queries (used by MultimodalManager)
    top_k_images: int = 3

    # Number of image results returned by the retrieve_image tool
    # (1 is sufficient — index now contains focused diagram crops, not full pages)
    top_k_images_retrieve: int = 1

    # ── API ──────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file="../.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached Settings singleton.

    Using lru_cache means .env is only parsed once per process, not on
    every request. Call `get_settings.cache_clear()` in tests to reset.
    """
    return Settings()
