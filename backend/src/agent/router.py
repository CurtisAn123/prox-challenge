"""
src/agent/router.py
─────────────────────
THE BRAIN — The Agent Router.

Replaces the previous "Tool-as-Node" dispatch table with a
Reasoning-Driven Multimodal Architecture:

  User Query
      │
      ▼
  ┌─────────────────────────────────┐
  │  StrategicPlanner (Haiku)       │  Decompose query into ordered sub-tasks
  └──────────────┬──────────────────┘
                 │
                 ▼
  ┌─────────────────────────────────┐
  │  ReasoningLoop (Opus, ReAct)    │  Think → Act → Observe × N
  │  ┌─────────────────────────────┐│
  │  │ Tools available:            ││
  │  │  search_text                ││  ← VectorStore
  │  │  search_kg_entity           ││  ← KGInterface (fuzzy BFS)
  │  │  find_kg_path               ││  ← KGInterface (nx.shortest_path)
  │  │  retrieve_image             ││  ← ImageRetriever
  │  │  analyze_image_with_context ││  ← MultimodalManager (VLM)
  │  │  calculate_duty_cycle       ││  ← duty_cycle_tool
  │  │  generate_diagram           ││  ← DiagramGenerator
  │  │  finish                     ││  ← terminates loop
  │  └─────────────────────────────┘│
  └──────────────┬──────────────────┘
                 │
                 ▼
  Response envelope: {intent, type, sources, content}
  (identical shape to the previous architecture — frontend unchanged)

──────────────────────────────────────────────────────────────
  RESPONSE ENVELOPE
──────────────────────────────────────────────────────────────

  {
    "type":    "text" | "mermaid" | "image",
    "intent":  "reasoning" | "reasoning_timeout",
    "sources": list[dict],
    "content": dict
  }

  text    → content: {"answer": str}
  mermaid → content: {"syntax": str, "title": str}
  image   → content: {"images": list[dict]}
"""

import re
from typing import Any, Callable

import anthropic

from config.settings import get_settings
from src.agent.kg_interface import KGInterface
from src.agent.multimodal_manager import MultimodalManager
from src.agent.prompts import build_prompt, REASONING_SYSTEM_TEMPLATE
from src.agent.reasoning_loop import ReasoningLoop, LoopResult
from src.agent.strategic_planner import StrategicPlanner
from src.agent.tool_executor import ToolExecutor
from src.generators.diagram_generator import DiagramGenerator
from src.ingestion.graph_builder import KnowledgeGraph
from src.ingestion.vector_store import VectorStore
from src.retrieval.image_retriever import ImageRetriever
from src.schemas.widget_schemas import WidgetSchemaRegistry


class AgentRouter:
    """
    Orchestrates the StrategicPlanner → ReasoningLoop pipeline.

    Instantiated once at application startup (in api/main.py) with all
    pre-built dependencies injected.

    The ``vector_store`` and ``kg`` attributes are kept as direct references
    so the /health endpoint in api/routes/query.py can read them without
    changes.

    Usage::

        router = AgentRouter(
            vector_store=vs,
            knowledge_graph=kg,
            image_retriever=ir,
            diagram_generator=dg,
            widget_registry=wr,
            kg_interface=kgi,
            multimodal_manager=mm,
        )
        response = router.route("What polarity do I need for TIG welding?")
    """

    def __init__(
        self,
        vector_store: VectorStore,
        knowledge_graph: KnowledgeGraph,
        image_retriever: ImageRetriever,
        diagram_generator: DiagramGenerator,
        widget_registry: WidgetSchemaRegistry,
        kg_interface: KGInterface,
        multimodal_manager: MultimodalManager,
    ) -> None:
        # Kept as attributes for /health endpoint compatibility
        self.vector_store = vector_store
        self.kg = knowledge_graph
        self.widget_registry = widget_registry

        self.settings = get_settings()
        client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)

        tool_executor = ToolExecutor(
            vector_store=vector_store,
            kg_interface=kg_interface,
            image_retriever=image_retriever,
            multimodal_manager=multimodal_manager,
            diagram_generator=diagram_generator,
            settings=self.settings,
        )

        system_prompt = build_prompt(REASONING_SYSTEM_TEMPLATE, self.settings.product_name)

        self.planner = StrategicPlanner(
            client=client,
            fast_model=self.settings.claude_fast_model,
            product_name=self.settings.product_name,
        )

        self.loop = ReasoningLoop(
            anthropic_client=client,
            reasoning_model=self.settings.claude_model,
            tool_executor=tool_executor,
            multimodal_manager=multimodal_manager,
            product_name=self.settings.product_name,
            system_prompt=system_prompt,
        )

    # ── Public entry point ────────────────────────────────────────────────────

    def route(
        self,
        query: str,
        status_callback: Callable[[str, dict], None] | None = None,
    ) -> dict[str, Any]:
        """
        Main entry point. Accept a raw user query and return a typed response.

        Steps:
          1. StrategicPlanner decomposes query → Plan (Haiku, cheap).
          2. ReasoningLoop executes ReAct cycle → LoopResult (Opus).
          3. LoopResult is wrapped in the standard response envelope.

        *status_callback* is forwarded to the ReasoningLoop so the streaming
        endpoint can emit progress events to the frontend.
        """
        query = query.strip()
        if status_callback:
            status_callback("planning", {})
        plan = self.planner.plan(query)
        result = self.loop.run(query, plan, status_callback=status_callback)
        return {
            "intent": result.intent_label,
            "type":   result.response_type,
            "sources": result.sources,
            "content": _build_content(result, self.widget_registry),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_content(result: LoopResult, widget_registry: WidgetSchemaRegistry) -> dict[str, Any]:
    """Map a LoopResult to the content dict the frontend expects."""
    if result.response_type == "mermaid":
        return {"syntax": result.mermaid_syntax or "", "title": result.answer}

    if result.response_type == "image":
        return {"answer": result.answer, "images": result.images or []}

    if result.response_type == "rich":
        # Composite: prose explanation + optional Mermaid schematic + optional manual images
        content: dict[str, Any] = {"answer": result.answer, "images": result.images or []}
        if result.mermaid_syntax:
            _title_match = re.search(r"%%\s*Title:\s*(.+)", result.mermaid_syntax)
            _mermaid_title = _title_match.group(1).strip() if _title_match else "Diagram"
            content["mermaid"] = {"syntax": result.mermaid_syntax, "title": _mermaid_title}
        return content

    if result.response_type == "widget":
        # LLM selected a widget key directly — use direct key lookup (not keyword matching)
        wd = result.widget_data or {}
        widget_content = widget_registry.get_widget_by_key(
            widget_key=wd.get("widget_key", "wire_speed"),
            pre_populated=wd.get("pre_populated"),
        )
        widget_content["summary"] = result.answer
        return widget_content

    # Default: text
    return {"answer": result.answer}
