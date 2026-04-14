# Frontend

React + Vite chat UI for the Vulcan OmniPro 220 agent.

---

## Layout

```
frontend/
├── src/
│   ├── App.jsx               Root component — layout, dark mode, health polling
│   ├── main.jsx              React entry point
│   ├── index.css             Tailwind base + custom CSS variables
│   ├── api/
│   │   └── client.js         streamQuery() (SSE), getHealth()
│   ├── hooks/
│   │   └── useChat.js        Chat state — messages, loading, submit, clear
│   └── components/
│       ├── ChatInput.jsx     Message input + submit button
│       ├── MessageList.jsx   Scrollable message thread
│       ├── Message.jsx       Single message — routes to the right renderer
│       ├── SourceCitations.jsx  Collapsible source citations panel
│       └── renderers/
│           ├── TextRenderer.jsx    Markdown prose
│           ├── MermaidRenderer.jsx Mermaid diagram (live rendered)
│           ├── ImageRenderer.jsx   Manual images with captions
│           └── WidgetRenderer.jsx  Interactive configurator forms
├── index.html
├── vite.config.js
├── tailwind.config.js
└── postcss.config.js
```

---

### Mermaid

Mermaid renders diagrams from the agent's `mermaid` response type. When the backend returns `{ type: "mermaid", content: { syntax, title } }`, `MermaidRenderer` calls `mermaid.render()` with the syntax string and injects the resulting SVG.

**Why Mermaid:** The agent generates diagram syntax dynamically at inference time. It outputs the Mermaid source as text. A charting library would require structured data (nodes, edges, series) parsed out of the model output and re-serialized into a component API. Mermaid accepts a text string directly, which is the natural output format for the agent. The agent can generate a troubleshooting flowchart, a wiring schematic, or a process selection diagram without the frontend knowing the shape of the data in advance.

### SSE streaming (no WebSocket)

The `/query` endpoint streams progress events over Server-Sent Events. The frontend uses `EventSource`-style streaming via `fetch` with a `ReadableStream` reader. Can implement streaming in the future for more transparency for the user.

### Source citations

Every response includes a `sources` array from the backend that lists page numbers, section headers, relevance scores, and image captions from the retrieval step.

---

## Running

```bash
cd frontend
npm install
npm run dev        # Vite dev server on http://localhost:5173
```

The dev server proxies `/api/*` to `http://localhost:8000` (set in `vite.config.js`). Start the backend first.

```bash
npm run build      # Production build to dist/
npm run preview    # Preview production build locally
```

---

## Environment

The only frontend env variable is `VITE_API_URL`, which defaults to `/api` (proxied by Vite in dev, same-origin in production). Set it in `.env` at the repo root if you're pointing the frontend at a remote backend:

```
VITE_CONFIG_URL -> http://your-backend-host:8000
```
