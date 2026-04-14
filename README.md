# Vulcan OmniPro 220 — Expert AI Assistant

<img src="product.webp" alt="Vulcan OmniPro 220" width="400" /> <img src="product-inside.webp" alt="Vulcan OmniPro 220 — inside panel" width="400" />


## Quick start

### Environment Variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | required | Anthropic API key |
| `MISTRAL_API_KEY` | required | Mistral API key (for OCR ingestion) |
| `MISTRAL_API_MODEL` | required | mistral-ocr-latest |

Create mistral_api_key here: https://console.mistral.ai/home

```bash
git clone <your-fork>
cd <your-fork>

cp .env.example .env
# Fill in ANTHROPIC_API_KEY and MISTRAL_API_KEY in .env

# Backend
cd backend
uv venv && uv pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000

# In a second terminal — Frontend
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

On first run, the backend boots into "ingestion-required" mode. Hit the **Ingest** button in the UI (or `POST /ingest`) to build the knowledge base. This takes a bit but is only needed once. The directory already has the sectioned images and knowledge graph built. To run a full reset (at the cost of your API credits), remove ALL data from /data/chroma/, ./images/, and kowledge_graph.pkl. Then run ingest again.

---

## What it can do

- **Technical lookups** — duty cycles, voltage/amperage tables, wire feed specs
- **Visual answers** — surfaces manual diagrams inline when they're more useful than text
- **Generated diagrams** — draws Mermaid flowcharts and schematics on the fly for wiring, process selection, troubleshooting paths
- **Interactive widgets** — wire speed / voltage configurator that takes process + material + thickness and outputs recommended settings
- **Cross-referenced reasoning** — the ReAct loop can call multiple tools and synthesize across them before answering

---


See [backend/README.md](backend/README.md) and [frontend/README.md](frontend/README.md) for library choices, design decisions, and detailed documentation.
