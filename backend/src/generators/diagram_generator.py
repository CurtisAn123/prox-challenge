"""
src/generators/diagram_generator.py
─────────────────────────────────────
Generates Mermaid.js diagram syntax for spatial / connection questions.

When a user asks "what polarity do I need for TIG?" or "how does the wire
feed path work?", a text answer is insufficient. This generator:

  1. Retrieves the relevant context via VectorStore.
  2. Asks Claude to produce a Mermaid diagram that visually encodes the answer.
  3. Returns the raw Mermaid syntax — the frontend renders it with mermaid.js.

Supported diagram types:
  - flowchart: decision trees, process steps, troubleshooting flows
  - graph LR: physical connections (polarity wiring, cable routing)
  - sequenceDiagram: multi-step setup procedures
  - classDiagram: component hierarchies

The type is chosen automatically based on the query content.
"""

import re
from typing import Any

import anthropic

from src.ingestion.vector_store import VectorStore
from config.settings import get_settings


# ──────────────────────────────────────────────────────────────────────────────
#  Diagram generation prompt
# ──────────────────────────────────────────────────────────────────────────────

DIAGRAM_SYSTEM_PROMPT = """\
You are a technical diagramming expert for the Vulcan OmniPro 220 multiprocess welder.

Given a user's troubleshooting problem and relevant manual passages, generate a
Mermaid.js decision-tree flowchart that guides the user step-by-step to diagnose
and resolve the issue.

RULES:
1. Always use "flowchart TD" (top-down).
2. Structure as a single vertical spine of checks — one column, top to bottom:
   - Start node: the symptom/problem (rectangle).
   - Check nodes (diamond): one yes/no question the user can answer.
   - Fix nodes (rectangle): what to do when a check fails.
   - CRITICAL: every fix node MUST reconnect to the next check node with an arrow.
     This keeps the layout in one column. Dead-end fix nodes cause horizontal sprawl.
   - End node: "Issue resolved / seek further help" at the bottom.
3. Use CLEAR, SHORT node labels. Never use \\n inside node labels — keep each label on a single line.
4. Style nodes by role:
   - style NODEID fill:#fee2e2,stroke:#ef4444  (problem/symptom node — red)
   - style NODEID fill:#fef3c7,stroke:#f59e0b  (check/decision node — amber)
   - style NODEID fill:#e8f5e9,stroke:#4CAF50  (fix/resolution node — green)
5. Include a title comment at the top: %% Title: <diagram title>
6. Return ONLY the Mermaid code — no markdown, no explanation.

Example output for a porosity problem (note how every fix node Ax reconnects to the next check Qx+1):
---
%% Title: Diagnosing Weld Porosity
flowchart TD
    START["Weld has porosity - holes or pits"]
    Q1{"Is base metal clean and dry?"}
    A1["Clean with wire brush and degrease"]
    Q2{"Is polarity set to DCEN?"}
    A2["Switch polarity to DCEN - electrode negative"]
    Q3{"Is gas flow rate correct - 20-25 CFH?"}
    A3["Adjust regulator to 20-25 CFH"]
    DONE["Checks complete - retest weld"]
    START --> Q1
    Q1 -->|"No"| A1
    A1 --> Q2
    Q1 -->|"Yes"| Q2
    Q2 -->|"No"| A2
    A2 --> Q3
    Q2 -->|"Yes"| Q3
    Q3 -->|"No"| A3
    A3 --> DONE
    Q3 -->|"Yes"| DONE
    style START fill:#fee2e2,stroke:#ef4444
    style Q1 fill:#fef3c7,stroke:#f59e0b
    style Q2 fill:#fef3c7,stroke:#f59e0b
    style Q3 fill:#fef3c7,stroke:#f59e0b
    style A1 fill:#e8f5e9,stroke:#4CAF50
    style A2 fill:#e8f5e9,stroke:#4CAF50
    style A3 fill:#e8f5e9,stroke:#4CAF50
    style DONE fill:#e8f5e9,stroke:#4CAF50
---
"""

DIAGRAM_USER_TEMPLATE = """\
Question: {query}

Manual context:
{context}

Generate a Mermaid diagram that directly answers this question.
"""


# ──────────────────────────────────────────────────────────────────────────────
#  DiagramGenerator
# ──────────────────────────────────────────────────────────────────────────────

class DiagramGenerator:
    """
    Generates Mermaid.js diagrams on demand for spatial/connection queries.
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store
        self.settings = get_settings()
        self.client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)

    def generate(self, query: str) -> dict[str, Any]:
        """
        Generate a Mermaid diagram for *query*.

        Returns::
            {
                "type": "mermaid",
                "syntax": str,          # raw Mermaid.js syntax
                "title": str,           # extracted from the %% Title comment
                "context_sources": list # source citations used
            }
        """
        # Retrieve relevant context
        text_results = self.vector_store.search_text(query, top_k=4)
        context = "\n\n".join(r["text"] for r in text_results)
        sources = [
            {"source": r["source"], "page": r["page"]}
            for r in text_results
        ]

        # Generate diagram via Claude
        response = self.client.messages.create(
            model=self.settings.claude_model,
            max_tokens=1500,
            system=DIAGRAM_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": DIAGRAM_USER_TEMPLATE.format(
                        query=query, context=context[:3000]
                    ),
                }
            ],
        )

        raw = response.content[0].text.strip()

        # Strip markdown code fences if present (```mermaid ... ```)
        mermaid_syntax = re.sub(r"^```(?:mermaid)?\s*\n?", "", raw)
        mermaid_syntax = re.sub(r"\n?```$", "", mermaid_syntax).strip()

        # Extract title from comment
        title = "Diagram"
        title_match = re.search(r"%%\s*Title:\s*(.+)", mermaid_syntax)
        if title_match:
            title = title_match.group(1).strip()

        return {
            "type": "mermaid",
            "syntax": mermaid_syntax,
            "title": title,
            "context_sources": sources,
        }
