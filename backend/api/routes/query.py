"""
api/routes/query.py
─────────────────────
The primary query endpoint.

  POST /query  — accept a user question, return a typed multimodal response
  GET  /health — system health check (vector store counts, graph stats)
"""

import asyncio
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Any

router = APIRouter(tags=["query"])


# ──────────────────────────────────────────────────────────────────────────────
#  Request / Response models
# ──────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000, description="User's question")
    # Optional: force a specific route (useful for testing)
    force_intent: str | None = Field(
        None,
        description="Force routing to: text_qa | diagram | image | widget",
    )


class QueryResponse(BaseModel):
    """
    Typed response envelope.

    The ``type`` field tells the frontend which component to render:
      - "text"   → render as markdown
      - "image"  → render <img> tags with base64 src
      - "rich"   → render text + optional Mermaid diagram + optional images
      - "widget" → render the specified React component with schema props
    """
    intent: str
    type: str
    sources: list[dict]
    content: dict[str, Any]


# ──────────────────────────────────────────────────────────────────────────────
#  Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest):
    """
    Route a user question through the agent and return a multimodal response.

    The agent_router is attached to the FastAPI app state at startup.
    If the vector store is not yet populated (ingestion hasn't run), we
    return a helpful error rather than an empty/wrong response.
    """
    agent_router = request.app.state.agent_router

    if agent_router is None:
        raise HTTPException(
            503,
            "Agent not ready. Run POST /ingest first to populate the knowledge base.",
        )

    try:
        result = agent_router.route(body.query)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(500, f"Agent error: {str(e)}")


@router.post("/query/stream")
async def query_stream(request: Request, body: QueryRequest):
    """
    Streaming variant of POST /query.

    Emits Server-Sent Events (SSE) as the agent works, then a final
    ``done`` event carrying the full QueryResponse payload.

    Event shapes::

        data: {"type": "planning"}
        data: {"type": "thinking"}
        data: {"type": "tool_call", "tool": "<tool_name>"}
        data: {"type": "done", "result": { ...QueryResponse... }}
        data: {"type": "error", "message": "<msg>"}
    """
    agent_router = request.app.state.agent_router

    if agent_router is None:
        raise HTTPException(
            503,
            "Agent not ready. Run POST /ingest first to populate the knowledge base.",
        )

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    _DONE = object()

    def status_callback(event_type: str, data: dict) -> None:
        """Called from the worker thread — bridges into the async queue."""
        event = json.dumps({"type": event_type, **data})
        loop.call_soon_threadsafe(queue.put_nowait, event)

    async def run_agent() -> None:
        try:
            result = await loop.run_in_executor(
                None,
                lambda: agent_router.route(body.query, status_callback=status_callback),
            )
            await queue.put(json.dumps({"type": "done", "result": result}))
        except Exception as exc:
            await queue.put(json.dumps({"type": "error", "message": str(exc)}))
        finally:
            await queue.put(_DONE)

    async def generate():
        asyncio.create_task(run_agent())
        while True:
            item = await queue.get()
            if item is _DONE:
                break
            yield f"data: {item}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/health")
async def health(request: Request):
    """
    Return system health and readiness information.
    Useful for the frontend to show whether the agent is ready to answer.
    """
    agent_router = request.app.state.agent_router

    if agent_router is None:
        return {
            "status": "not_ready",
            "message": "Run POST /ingest to build the knowledge base.",
            "vector_store": {"text_chunks": 0, "images": 0},
            "graph": {},
        }

    vs = agent_router.vector_store
    kg = agent_router.kg

    return {
        "status": "ready",
        "vector_store": {
            "text_chunks": vs.text_collection.count(),
            "images": vs.image_collection.count(),
        },
        "graph": kg.summary(),
    }
