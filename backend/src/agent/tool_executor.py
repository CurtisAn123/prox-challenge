"""
src/agent/tool_executor.py
───────────────────────────
Dispatches Anthropic tool_use blocks to their Python implementations.

The ReasoningLoop calls ``execute(tool_name, tool_input)`` for every tool_use
block it receives. All results are wrapped in a uniform envelope:

    {"ok": True,  "result": <tool-specific dict>}   — success
    {"ok": False, "error":  <error string>}           — any failure

This envelope means the LLM always receives a parseable JSON object; error
strings are designed to be actionable so the model can self-correct.

The ``finish`` tool is intentionally NOT handled here — it is intercepted by
ReasoningLoop before dispatch to terminate the loop.
"""

from __future__ import annotations

import json
from typing import Any

from config.settings import Settings
from src.agent.kg_interface import KGInterface
from src.agent.multimodal_manager import MultimodalManager
from src.generators.diagram_generator import DiagramGenerator
from src.ingestion.vector_store import VectorStore
from src.retrieval.image_retriever import ImageRetriever
from src.tools.duty_cycle_tool import DUTY_CYCLE_TOOL_SCHEMA  # noqa: F401 — re-exported for REASONING_TOOLS


class ToolExecutor:
    """
    Central dispatcher for all agent tools.

    Parameters
    ----------
    vector_store       : VectorStore        — text chunk search
    kg_interface       : KGInterface        — knowledge graph queries
    image_retriever    : ImageRetriever     — image retrieval
    multimodal_manager : MultimodalManager  — VLM image analysis
    diagram_generator  : DiagramGenerator   — Mermaid diagram generation
    settings           : Settings           — top_k_text / top_k_images
    """

    def __init__(
        self,
        vector_store: VectorStore,
        kg_interface: KGInterface,
        image_retriever: ImageRetriever,
        multimodal_manager: MultimodalManager,
        diagram_generator: DiagramGenerator,
        settings: Settings,
    ) -> None:
        self.vs = vector_store
        self.kg = kg_interface
        self.image_retriever = image_retriever
        self.mm = multimodal_manager
        self.diagram_gen = diagram_generator
        self.settings = settings

        self._dispatch: dict[str, Any] = {
            "search_text":                 self._search_text,
            "search_kg_entity":            self._search_kg_entity,
            "find_kg_path":                self._find_kg_path,
            "retrieve_image":              self._retrieve_image,
            "analyze_image_with_context":  self._analyze_image_with_context,
            "calculate_duty_cycle":        self._calculate_duty_cycle,
            "generate_diagram":            self._generate_diagram,
        }

    # ── Public API ─────────────────────────────────────────────────────────────

    def execute(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a tool by name and return a uniform result envelope.

        Catches all exceptions so a single bad tool call never crashes the loop.
        """
        handler = self._dispatch.get(tool_name)
        if handler is None:
            return {"ok": False, "error": f"Unknown tool '{tool_name}'."}

        try:
            result = handler(**tool_input)
            return {"ok": True, "result": result}
        except TypeError as exc:
            # Malformed input (wrong/missing parameters from the LLM)
            return {"ok": False, "error": f"Invalid parameters for '{tool_name}': {exc}"}
        except Exception as exc:
            return {"ok": False, "error": f"Tool '{tool_name}' failed: {exc}"}

    # ── Tool handlers ──────────────────────────────────────────────────────────

    def _search_text(self, query: str) -> dict[str, Any]:
        """
        Semantic search over manual text chunks.

        Returns the top-k chunks with their text, score, source, page,
        and section. The LLM reads these as observations.
        """
        raw = self.vs.search_text(query, top_k=self.settings.top_k_text)
        chunks = [
            {
                "text": item["text"],
                "score": round(item["score"], 3),
                "source": item["source"],
                "page": item["page"],
                "section": item.get("section", ""),
            }
            for item in raw
        ]
        return {"chunks": chunks, "count": len(chunks)}

    def _search_kg_entity(self, entity_name: str) -> dict[str, Any]:
        """Look up an entity in the knowledge graph."""
        return self.kg.search_entity(entity_name)

    def _find_kg_path(self, from_entity: str, to_entity: str) -> dict[str, Any]:
        """Find the shortest relationship path between two KG entities."""
        return self.kg.find_path(from_entity, to_entity)

    def _retrieve_image(self, query: str) -> dict[str, Any]:
        """
        Retrieve images matching *query* by caption similarity.

        Returns a list of image dicts (base64_data, caption, page, score).
        """
        images = self.image_retriever.retrieve(query, top_k=self.settings.top_k_images_retrieve)
        return {
            "images": [img.to_dict() for img in images],
            "count": len(images),
        }

    def _analyze_image_with_context(
        self, figure_ref: str, context_text: str
    ) -> dict[str, Any]:
        """Send a retrieved image to the VLM for analysis."""
        result = self.mm.analyze(figure_ref, context_text)
        if result.get("analysis"):
            result["analysis"] = (
                result["analysis"]
                + "\n\n[Use the above analysis to inform your answer — do not copy it into finish().]"
            )
        return result

    def _calculate_duty_cycle(
        self, process: str, input_voltage: str, output_amps: int
    ) -> dict[str, Any]:
        """
        Retrieve duty cycle specifications from the KG and manual.

        Queries both the knowledge graph (DUTY CYCLE entity neighbors,
        RANGE_IS and DEPENDS_ON predicate facts) and the vector store
        (two targeted queries: one process/voltage/amps-specific, one
        for the general rated table). Returns all retrieved context for
        the LLM to read and synthesise into an answer.
        """
        # 1. KG: 1-hop neighborhood of DUTY CYCLE entity
        kg_entity = self.kg.search_entity("DUTY CYCLE")

        # 2. KG: RANGE_IS facts (rated duty-cycle values at specific amperages)
        kg_ranges = self.kg.query_by_predicate("RANGE_IS", subject="DUTY CYCLE")

        # 3. KG: DEPENDS_ON facts (what duty cycle depends on)
        kg_deps = self.kg.query_by_predicate("DEPENDS_ON", subject="DUTY CYCLE")

        # 4. Vector: specific process/voltage/amps query
        v_specific = self.vs.search_text(
            f"duty cycle {process} {input_voltage} {output_amps}A percentage rated"
        )

        # 5. Vector: general rated table query
        v_table = self.vs.search_text(
            "duty cycle table percentage amperage output rated breakpoints"
        )

        # Deduplicate by page number, cap at 6 excerpts
        seen_pages: set[int] = set()
        excerpts: list[dict[str, Any]] = []
        for item in v_specific + v_table:
            pg = item.get("page", 0)
            if pg not in seen_pages:
                seen_pages.add(pg)
                excerpts.append(item)
            if len(excerpts) >= 6:
                break

        return {
            "query": {
                "process": process,
                "input_voltage": input_voltage,
                "output_amps": output_amps,
            },
            "kg_neighbors": kg_entity.get("neighbors", []),
            "kg_range_facts": kg_ranges.get("results", []),
            "kg_dependency_facts": kg_deps.get("results", []),
            "manual_excerpts": excerpts,
            "source_pages": sorted(seen_pages),
        }

    def _generate_diagram(self, query: str) -> dict[str, Any]:
        """
        Generate a Mermaid.js diagram for wiring/connection queries.

        Returns: {syntax: str, title: str, context_sources: list}
        """
        return self.diagram_gen.generate(query)


# ──────────────────────────────────────────────────────────────────────────────
#  Tool schemas (Anthropic tool-use API format)
#
#  REASONING_TOOLS is the list passed to client.messages.create(tools=...).
#  The finish tool is included here because it must appear in the API call
#  even though the loop intercepts it before ToolExecutor.execute().
# ──────────────────────────────────────────────────────────────────────────────

REASONING_TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_text",
        "description": (
            "Semantic search over the product manual's text chunks. "
            "Returns the most relevant passages for a given query. "
            "Always use this to retrieve factual content before answering. "
            "Check returned text for figure/diagram references and follow up "
            "with analyze_image_with_context if deeper visual analysis is needed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Specific search query using technical terms from the domain.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_kg_entity",
        "description": (
            "Look up an entity in the knowledge graph and return its 1-hop "
            "neighborhood — all relationships it has with other entities. "
            "Use this BEFORE search_text for questions about components, "
            "processes, faults, or settings. The KG gives you structured "
            "facts (REQUIRES, CAUSED_BY, FIXED_BY, DEPENDS_ON, etc.) that "
            "are faster and more precise than full-text search. "
            "Fuzzy matching is applied so approximate names are acceptable."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_name": {
                    "type": "string",
                    "description": (
                        "Entity to look up. Use uppercase noun phrases, e.g. "
                        "'MIG WELDING', 'WIRE FEED SPEED', 'POROSITY', "
                        "'SHIELDING GAS'. Exact case not required."
                    ),
                }
            },
            "required": ["entity_name"],
        },
    },
    {
        "name": "find_kg_path",
        "description": (
            "Find the shortest relationship path between two knowledge graph "
            "entities. Use for causal or dependency questions: "
            "'Why does X cause Y?', 'How does A relate to B?', "
            "'What connects X to Y?'. Returns each hop with its predicate "
            "(e.g. CAUSED_BY, REQUIRES, FIXED_BY) and source page."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "from_entity": {
                    "type": "string",
                    "description": "Starting entity (uppercase noun phrase).",
                },
                "to_entity": {
                    "type": "string",
                    "description": "Target entity (uppercase noun phrase).",
                },
            },
            "required": ["from_entity", "to_entity"],
        },
    },
    {
        "name": "retrieve_image",
        "description": (
            "Search for images or diagrams relevant to the query and return "
            "them as base64-encoded PNG with captions and page references. "
            "Use when the user asks to SEE something, or when a visual "
            "would meaningfully improve the answer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Visual description of what you are looking for.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "analyze_image_with_context",
        "description": (
            "Retrieve the image best matching figure_ref and send it to a "
            "vision model for detailed analysis, grounded by context_text. "
            "Use when retrieved text passages describe or reference a visual "
            "element that would add important detail to the answer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "figure_ref": {
                    "type": "string",
                    "description": (
                        "Description of the figure to retrieve, e.g. "
                        "'front panel controls', 'wire feed mechanism', "
                        "'polarity wiring diagram'. Used as the image search query."
                    ),
                },
                "context_text": {
                    "type": "string",
                    "description": (
                        "The text passage that references or describes this image. "
                        "Passed to the vision model as grounding context."
                    ),
                },
            },
            "required": ["figure_ref", "context_text"],
        },
    },
    DUTY_CYCLE_TOOL_SCHEMA,
    {
        "name": "generate_diagram",
        "description": (
            "Generate a Mermaid.js decision-tree flowchart for complex multi-step "
            "troubleshooting questions. Use ONLY when the user describes a problem "
            "that requires several diagnostic steps — e.g. porosity, arc issues, "
            "burn-through, spatter. Do NOT use for simple polarity or wiring questions. "
            "Returns Mermaid syntax — always call finish(type='mermaid') after."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The troubleshooting problem to build a diagnostic flowchart for.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "return_widget",
        "description": (
            "Return an interactive configurator widget as the final response. "
            "Use INSTEAD of finish() for these query types: "
            "  wire_speed — user asks for wire feed speed or voltage for a material/thickness; "
            "  troubleshooting — user describes a weld defect or symptom to diagnose; "
            "  process_selector — user asks which welding process to use. "
            "The widget lets the user fill in parameters and get computed recommendations. "
            "DO NOT use for duty-cycle questions — those use calculate_duty_cycle + finish."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "widget_key": {
                    "type": "string",
                    "enum": ["wire_speed", "troubleshooting", "process_selector"],
                    "description": (
                        "Which widget to render. "
                        "wire_speed: settings for material/thickness; "
                        "troubleshooting: fault diagnosis flowchart; "
                        "process_selector: process recommendation wizard."
                    ),
                },
                "summary": {
                    "type": "string",
                    "description": (
                        "Brief text (1–2 sentences) shown above the widget explaining "
                        "why this configurator is the right tool. "
                        "Grounded in retrieved manual content."
                    ),
                },
                "pre_populated": {
                    "type": "object",
                    "description": (
                        "Optional values to pre-fill in the widget form, extracted from "
                        "the user's query. Keys must match the widget's field names exactly. "
                        "Example: {\"material\": \"Mild Steel\", \"thickness\": '1/4\"'}."
                    ),
                },
            },
            "required": ["widget_key", "summary"],
        },
    },
    {
        "name": "finish",
        "description": (
            "Terminate the reasoning loop and return the final answer. "
            "ALWAYS call this as the last action (unless using return_widget) — "
            "never produce a plain text response without calling finish. "
            "For text answers:      type='text', answer='...' "
            "For responses with a diagram: type='rich', answer='<explanation>', mermaid_syntax='...' "
            "For images only:       type='image', answer='<full text answer pointing to diagram location>' "
            "Do NOT use type='mermaid' — it suppresses the answer text entirely. "
            "Do NOT include image data — retrieved images are attached automatically."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": (
                        "Final answer in markdown. Always a direct, meaningful response "
                        "synthesized from your research — never a copy of the VLM analysis. "
                        "For rich (diagram): 1–4 sentences or bullets addressing the question. "
                        "For image: 1–3 sentences pointing to the relevant diagram location."
                    ),
                },
                "type": {
                    "type": "string",
                    "enum": ["text", "image", "rich"],
                    "description": (
                        "Response type. Use 'rich' whenever your response includes a Mermaid "
                        "diagram — it shows text + diagram together. Never use 'mermaid' alone "
                        "(it suppresses the answer text entirely, showing only the diagram)."
                    ),
                },
                "mermaid_syntax": {
                    "type": "string",
                    "description": "Required when type='mermaid' or type='rich'. Raw Mermaid.js syntax string.",
                },
            },
            "required": ["answer", "type"],
        },
    },
]
