"""
api/main.py
─────────────
FastAPI application entry point.

Startup sequence:
  1. Load settings from .env
  2. Check whether ChromaDB is already populated (fast path on restarts)
  3. If populated, instantiate all retrieval components and wire up the router
  4. If not populated, boot in "ingestion-required" mode
     (agent_router = None; /ingest must be called first)

Run locally:
  uvicorn api.main:app --reload --port 8000

The frontend should CORS-preflight to http://localhost:8000.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config.settings import get_settings


# ──────────────────────────────────────────────────────────────────────────────
#  Application lifespan
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup and once at shutdown.

    On startup we try to boot the agent immediately if the vector store
    is already populated from a previous run. This makes restarts fast.
    """
    settings = get_settings()
    app.state.agent_router = None  # default: not ready

    try:
        print("── Booting Vulcan OmniPro 220 QA Agent ──────────────────────")
        app.state.agent_router = _build_router(settings)
        print("── Agent ready ───────────────────────────────────────────────")
    except Exception as e:
        print(f"[WARN] Could not auto-boot agent: {e}")
        print("       Run POST /ingest to build the knowledge base.")

    yield  # Application runs here

    # Shutdown: nothing to clean up (ChromaDB is persistent, no open sockets)
    print("── Shutting down ─────────────────────────────────────────────────")


def _build_router(settings):
    """
    Instantiate all components and wire them into an AgentRouter.

    This function is called both at startup (if already populated) and
    after ingestion completes (to refresh the router with new data).
    """
    import anthropic
    from src.ingestion.vector_store import VectorStore
    from src.ingestion.graph_builder import KnowledgeGraph
    from src.retrieval.image_retriever import ImageRetriever
    from src.generators.diagram_generator import DiagramGenerator
    from src.schemas.widget_schemas import WidgetSchemaRegistry
    from src.agent.kg_interface import KGInterface
    from src.agent.multimodal_manager import MultimodalManager
    from src.agent.router import AgentRouter

    # ── Vector store ──────────────────────────────────────────────────────────
    vs = VectorStore()
    if not vs.is_populated():
        raise RuntimeError("Vector store is empty. Run POST /ingest first.")

    # ── Knowledge graph ───────────────────────────────────────────────────────
    kg = KnowledgeGraph()
    if not kg.load():
        # Graph doesn't exist yet — boot without it (graceful degradation)
        print("[WARN] No knowledge graph found. KG-based reasoning will be limited.")

    # ── Reasoning-layer components ────────────────────────────────────────────
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    kg_interface = KGInterface(kg)

    # ── Retrieval / generation components ────────────────────────────────────
    image_retriever = ImageRetriever(vs)
    diagram_gen = DiagramGenerator(vs)
    widget_registry = WidgetSchemaRegistry(vs, kg_interface)
    multimodal_manager = MultimodalManager(
        image_retriever=image_retriever,
        anthropic_client=client,
        vision_model=settings.claude_vision_model,
    )

    # ── Assemble the router ───────────────────────────────────────────────────
    return AgentRouter(
        vector_store=vs,
        knowledge_graph=kg,
        image_retriever=image_retriever,
        diagram_generator=diagram_gen,
        widget_registry=widget_registry,
        kg_interface=kg_interface,
        multimodal_manager=multimodal_manager,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  FastAPI app
# ──────────────────────────────────────────────────────────────────────────────

def get_router():
    """Return the agent router from app state (used by background ingest task)."""
    return app.state.agent_router


settings = get_settings()

app = FastAPI(
    title="Vulcan OmniPro 220 QA Agent",
    description=(
        "Multimodal RAG agent for the Vulcan OmniPro 220 multiprocess welder. "
        "Answers technical questions with text, diagrams, images, and interactive widgets."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Allow the frontend (on any port during development) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
from api.routes.query import router as query_router
from api.routes.ingest import router as ingest_router

app.include_router(query_router)
app.include_router(ingest_router)

# ── Static files (serve extracted images) ────────────────────────────────────
images_dir = Path("./data/images")
images_dir.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(images_dir)), name="images")


# ──────────────────────────────────────────────────────────────────────────────
#  Dev entrypoint
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
