# Backend

FastAPI service that runs the multimodal RAG agent for the Vulcan OmniPro 220.

---

## Layout

```
backend/
├── api/
│   ├── main.py           FastAPI app, lifespan, CORS, static files
│   └── routes/
│       ├── query.py      POST /query (streaming SSE), GET /health
│       └── ingest.py     POST /ingest (background task)
├── config/
│   └── settings.py       Pydantic-settings config, loaded from ../.env
├── src/
│   ├── agent/
│   │   ├── router.py           AgentRouter — top-level orchestrator
│   │   ├── strategic_planner.py StrategicPlanner — Haiku query decomposition
│   │   ├── reasoning_loop.py   ReasoningLoop — ReAct Think/Act/Observe cycle
│   │   ├── tool_executor.py    ToolExecutor — dispatches tool_use blocks
│   │   ├── kg_interface.py     KGInterface — fuzzy BFS and path queries over KG
│   │   ├── multimodal_manager.py MultimodalManager — VLM image analysis
│   │   └── prompts.py          System prompt templates
│   ├── ingestion/
│   │   ├── mistral_ocr.py      Mistral OCR PDF parser (primary)
│   │   ├── pdf_parser.py       PyMuPDF + pdfplumber (utilities, table extraction)
│   │   ├── vector_store.py     ChromaDB wrapper — text and image collections
│   │   └── graph_builder.py    NetworkX knowledge graph — triplet extraction + persistence
│   ├── retrieval/
│   │   ├── text_retriever.py   Semantic search over text chunks
│   │   └── image_retriever.py  Semantic search over image captions
│   ├── generators/
│   │   └── diagram_generator.py Mermaid diagram generation with RAG context
│   ├── schemas/
│   │   └── widget_schemas.py   Widget schema registry (duty cycle configurator, etc.)
│   └── tools/
│       └── duty_cycle_tool.py  Structured duty cycle lookup — exact values, no hallucination
└── data/                       Created at runtime
    ├── chroma/                 ChromaDB persistence
    ├── knowledge_graph.pkl     Serialised NetworkX graph
    └── images/                 Extracted PDF images (served as static files)
```

---

## Libraries and design choices

### FastAPI

The API layer is FastAPI with Uvicorn. The main reasons:

- **Async-native.** The agent reasoning loop makes multiple LLM calls in sequence. With async routes, those calls don't block the event loop.
- **Server-Sent Events (SSE).** The `/query` endpoint streams progress events (`planning`, `thinking`, `tool_call`) back to the frontend as the ReAct loop runs. FastAPI's `StreamingResponse` makes this straightforward without pulling in a WebSocket framework.
- **Startup lifespan.** FastAPI's `@asynccontextmanager` lifespan hook loads all agent components once at startup (not per-request). This avoids reconstructing ChromaDB connections and reloading the embedding model on every query.

### Anthropic Claude SDK — three model roles

Three Claude models are used for three distinct jobs:

| Model | Role | Why |
|-------|------|-----|
| `claude-haiku-4-5` | StrategicPlanner | Query decomposition is cheap classification work. Haiku is fast and costs a fraction of Opus |
| `claude-opus-4-6` | ReasoningLoop | The core ReAct cycle needs deep reasoning to correctly cross-reference duty cycle tables, understand wiring diagrams, and generate accurate answers |
| `claude-opus-4-6` | Vision (image captioning + VLM analysis) | Captioning accuracy at ingestion time directly determines retrieval quality. |

**ReAct over a single chain:** The reasoning loop is a standard ReAct (Reason + Act) implementation. The model calls tools, sees results, decides what else it needs, calls more tools, and signals done via `finish()`. This is significantly more accurate than a single-shot RAG call because the model can discover mid-loop that its initial retrieval was incomplete and issue a follow-up search.

**StrategicPlanner:** Complex questions benefit from decomposition before the loop starts. The planner (Haiku, fast) converts the question into an ordered plan that the loop receives as a preamble.

### Mistral OCR (`mistral-ocr-latest`)

PDF parsing is the most consequential decision in the ingestion pipeline (lowkey spent a lot of time here). The Vulcan manual is dense, mistral OCR gave the best results compared to other PDF parsers.

Only downside is that for production, may bear costs. For this POC, using the free API key was sufficient.

Created here: https://console.mistral.ai/home

Mistral owns detection; Claude owns interpretation. Mistral extracts what's on the page. Claude (vision model) captions the extracted image crops — what the component does, how to set it up, what failure mode it shows.

### ChromaDB

The vector store is ChromaDB running embedded (in-process, no server). Two collections:

- `text_chunks` — text passages with metadata (page, section, source)
- `image_captions` — Claude-generated captions with image file paths

ChromaDB chosen for it's easy set up for a POC.

### NetworkX knowledge graph

The knowledge graph encodes domain relationships extracted from the manual text:

- `Component → PART_OF → Assembly`
- `Process → REQUIRES → Setting / Component`
- `Fault → FIXED_BY → Troubleshooting step`
- `Setting → DEPENDS_ON → Material / Thickness / Voltage`

Triplets are extracted by Claude during ingestion (one call per chunk). The graph is serialised as a pickle and loaded at startup.

This helps link together all the components, so when the agent is determining what information to gather, it's able to traverse the graph to locate relevant chunks to form an answer.

**Why NetworkX over Neo4j:** NetworkX runs in-process with no infrastructure as  the graph for a 48-page manual is small. Neo4j would add operational complexity (a running server, a driver, a query language) for a dataset this size.

The KG is used as a supplement to vector search, not a replacement. When the ReAct loop calls `search_kg_entity`, it retrieves the 1-hop neighborhood of entities mentioned in the query. This provides relational context that dense vector search sometimes misses.

## Configuration

All settings are in [config/settings.py](config/settings.py) and loaded from `../.env` (repo root).

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | required | Anthropic API key |
| `MISTRAL_API_KEY` | required | Mistral API key (for OCR ingestion) |

---

## Running

```bash
cd backend
uv venv && source .venv/bin/activate   # or: python -m venv .venv
uv pip install -r requirements.txt

uvicorn api.main:app --reload --port 8000
```

On first run, `POST /ingest` must be called to build the knowledge base. After that, the agent auto-boots on every restart using the persisted ChromaDB and graph.

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | Stream a query through the agent (SSE) |
| `GET` | `/health` | Returns agent readiness + vector store stats |
| `POST` | `/ingest` | Trigger knowledge base build (background task) |
| `GET` | `/images/{filename}` | Serve extracted manual images |
