"""
api/routes/ingest.py
──────────────────────
Endpoints for triggering and monitoring the ingestion pipeline.

  POST /ingest        — parse PDFs, build graph, populate vector store
  GET  /ingest/status — report current ingestion state
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from api.main import get_router, app, _build_router  # shared router instance
from config.settings import get_settings

router = APIRouter(prefix="/ingest", tags=["ingestion"])

# Simple in-process state (replace with Redis for multi-worker deployments)
_ingest_status: dict = {"state": "idle", "message": "Not yet started."}


class IngestResponse(BaseModel):
    status: str
    message: str


@router.post("", response_model=IngestResponse)
async def trigger_ingest(background_tasks: BackgroundTasks):
    """
    Kick off the ingestion pipeline in the background.

    The pipeline:
      1. Parses all PDFs in the files/ directory.
      2. Captions images with Claude Vision.
      3. Embeds and stores text chunks in ChromaDB.
      4. Extracts knowledge graph triplets via Claude.

    This may take 5–15 minutes on first run depending on PDF size and
    how many images need to be captioned.
    """
    global _ingest_status

    if _ingest_status["state"] == "running":
        raise HTTPException(409, "Ingestion already in progress.")

    background_tasks.add_task(_run_ingestion)
    _ingest_status = {"state": "running", "message": "Pipeline started."}
    return IngestResponse(status="started", message="Ingestion pipeline started in background.")


@router.get("/status", response_model=IngestResponse)
async def ingest_status():
    """Return the current state of the ingestion pipeline."""
    return IngestResponse(
        status=_ingest_status["state"],
        message=_ingest_status["message"],
    )


async def _run_ingestion():
    """
    Background task that runs the full ingestion pipeline.
    Updates _ingest_status throughout so the /status endpoint reflects progress.
    """
    global _ingest_status
    try:
        from src.ingestion.mistral_ocr import MistralOCRParser
        from src.ingestion.vector_store import VectorStore
        from src.ingestion.graph_builder import KnowledgeGraph

        _ingest_status = {"state": "running", "message": "Parsing PDFs with Mistral OCR…"}
        parser = MistralOCRParser()
        chunks, images = parser.parse_all()

        _ingest_status = {"state": "running", "message": f"Parsed {len(chunks)} chunks and {len(images)} images. Embedding…"}
        vs = VectorStore()
        vs.add_text_chunks(chunks)

        _ingest_status = {"state": "running", "message": "Captioning and indexing images…"}
        vs.add_images(images)

        _ingest_status = {"state": "running", "message": "Building knowledge graph…"}
        kg = KnowledgeGraph()
        kg.build_from_chunks(chunks)

        app.state.agent_router = _build_router(get_settings())
        _ingest_status = {
            "state": "complete",
            "message": f"Done. {len(chunks)} chunks, {len(images)} images, graph built.",
        }

    except Exception as e:
        _ingest_status = {"state": "error", "message": str(e)}
        raise
