"""
src/retrieval/text_retriever.py
─────────────────────────────────
Retrieves text chunks relevant to a query and formats them for LLM context.

This module owns the "context window assembly" step of RAG:
  1. Semantic search via VectorStore
  2. Optional graph context injection via KnowledgeGraph
  3. Format into a single context string with source citations
"""

from typing import Any

from src.ingestion.vector_store import VectorStore
from src.ingestion.graph_builder import KnowledgeGraph
from config.settings import get_settings


class TextRetriever:
    """
    Assembles context strings for text-based QA.

    Combines vector-retrieved chunks with knowledge graph facts so the
    LLM has both verbatim manual passages AND structured relationships.
    """

    def __init__(self, vector_store: VectorStore, knowledge_graph: KnowledgeGraph) -> None:
        self.vector_store = vector_store
        self.kg = knowledge_graph
        self.settings = get_settings()

    def retrieve_context(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        """
        Build an augmented context dict for RAG.

        Returns::
            {
                "context_string": str,      # full context for LLM prompt injection
                "sources": list[dict],      # list of source citations
                "graph_facts": str,         # formatted graph neighborhood
            }
        """
        # 1. Vector similarity search
        text_results = self.vector_store.search_text(
            query, top_k=top_k or self.settings.top_k_text
        )

        # 2. Graph context (structured facts about entities in the query)
        graph_facts = self.kg.get_context_for_query(query)

        # 3. Assemble context string
        context_parts: list[str] = []

        if graph_facts:
            context_parts.append(graph_facts)
            context_parts.append("")  # blank line separator

        context_parts.append("Relevant passages from the Vulcan OmniPro 220 manual:")
        for i, result in enumerate(text_results, start=1):
            page_ref = f"[p.{result['page']}]" if result["page"] else ""
            section = f" — {result['section']}" if result["section"] else ""
            header = f"\n[Source {i}: {result['source']}{section} {page_ref}]"
            context_parts.append(f"{header}\n{result['text']}")

        sources = [
            {
                "source": r["source"],
                "page": r["page"],
                "section": r["section"],
                "score": round(r["score"], 3),
            }
            for r in text_results
        ]

        return {
            "context_string": "\n".join(context_parts),
            "sources": sources,
            "graph_facts": graph_facts,
        }
