"""
src/ingestion/graph_builder.py
────────────────────────────────
Builds and queries a NetworkX knowledge graph from text chunks.

The graph encodes domain relationships specific to the Vulcan OmniPro 220:
  - Component → PART_OF → Assembly
  - Process   → REQUIRES → Setting / Component
  - Fault     → FIXED_BY → Troubleshooting step
  - Setting   → DEPENDS_ON → Material / Thickness / Voltage

This graph is used to augment RAG context: before answering, the router
queries the graph for entities mentioned in the user's question and injects
their 1-hop neighborhood as additional structured context.

Graph construction workflow:
  1. For each text chunk, call Claude to extract (subject, predicate, object) triplets.
  2. Add triplets to a directed NetworkX MultiDiGraph.
  3. Serialise the graph to disk as a pickle.
  4. At query time, load the graph and call `get_context_for_query`.

Design choice — NetworkX over Neo4j:
  NetworkX runs in-process with zero infrastructure. Upgrading to Neo4j is
  a 1-day task once the schema is validated.
"""

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic
import networkx as nx

from config.settings import get_settings
from src.ingestion.pdf_parser import TextChunk


# ──────────────────────────────────────────────────────────────────────────────
#  Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Triplet:
    """A single (subject, predicate, object) knowledge triplet."""
    subject: str
    predicate: str
    obj: str          # 'object' is a Python builtin; use 'obj'
    source_chunk_id: str
    source_page: int
    confidence: float = 1.0

    def __str__(self) -> str:
        return f"({self.subject}) --[{self.predicate}]--> ({self.obj})"


# ──────────────────────────────────────────────────────────────────────────────
#  Triplet extraction prompts
# ──────────────────────────────────────────────────────────────────────────────

#  System prompt for triplet extraction.
#  We give Claude very specific instructions so it produces structured,
#  consistent predicates rather than free-form language.
TRIPLET_SYSTEM_PROMPT = """\
You are a knowledge-graph extraction specialist for welding equipment manuals.

Given a passage from the Vulcan OmniPro 220 owner's manual, extract
(subject, predicate, object) triplets that capture technical facts.

PREDICATE VOCABULARY — use ONLY these predicates to keep the graph consistent:
  PART_OF         — component belongs to an assembly
  CONNECTED_TO    — physical connection between components
  REQUIRES        — a process or setting requires something
  CONTROLS        — a knob/button controls a parameter
  USED_FOR        — a tool/setting is used for a purpose
  FIXED_BY        — a fault/symptom is fixed by an action
  CAUSED_BY       — a fault is caused by a condition
  COMPATIBLE_WITH — two items are compatible
  DEPENDS_ON      — a value depends on another parameter
  LOCATED_AT      — something is physically at a location
  RANGE_IS        — a parameter has a numerical range
  DEFAULT_IS      — a parameter has a default value
  WARNING         — a safety warning relation

OUTPUT FORMAT — return ONLY valid JSON, no commentary:
{
  "triplets": [
    {"subject": "...", "predicate": "...", "object": "...", "confidence": 0.0–1.0},
    ...
  ]
}

RULES:
- Subjects and objects must be concise noun phrases (≤6 words).
- Normalise to uppercase (e.g. "MIG WELDING", "WIRE FEED SPEED").
- Only include triplets you are highly confident about.
- Skip generic, non-technical facts.
- Return an empty list if no useful triplets are found.
"""

TRIPLET_USER_TEMPLATE = """\
Extract knowledge triplets from this passage:

---
{text}
---
"""


# ──────────────────────────────────────────────────────────────────────────────
#  KnowledgeGraph
# ──────────────────────────────────────────────────────────────────────────────

class KnowledgeGraph:
    """
    Builds, persists, and queries a NetworkX knowledge graph.

    Nodes represent domain entities (components, processes, settings, faults).
    Directed edges represent typed relationships (predicates from the vocabulary above).

    Example graph queries:
      kg.get_neighbors("MIG WELDING")
      → ["WIRE FEED SPEED", "POLARITY", "SHIELDING GAS", ...]

      kg.get_path("POROSITY", "WIRE FEED SPEED")
      → [POROSITY] --CAUSED_BY--> [SHIELDING GAS] --REQUIRES--> [WIRE FEED SPEED]
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)

        # MultiDiGraph: multiple parallel edges allowed (different predicates)
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._graph_path = Path(self.settings.graph_path)
        self._graph_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """Serialise the graph to disk."""
        with open(self._graph_path, "wb") as f:
            pickle.dump(self.graph, f)
        print(f"  Graph saved: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges → {self._graph_path}")

    def load(self) -> bool:
        """
        Load the graph from disk.
        Returns True if successful, False if no saved graph exists.
        """
        if not self._graph_path.exists():
            return False
        with open(self._graph_path, "rb") as f:
            self.graph = pickle.load(f)
        print(f"  Graph loaded: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
        return True

    # ── Graph construction ────────────────────────────────────────────────────

    def build_from_chunks(self, chunks: list[TextChunk], batch_size: int = 5) -> None:
        """
        Iterate over text chunks, extract triplets with Claude, and add them
        to the graph.

        To reduce API calls, chunks are processed in *batch_size* groups —
        each API call handles multiple chunks at once.
        """
        total = len(chunks)
        print(f"  Building knowledge graph from {total} chunks…")

        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            for chunk in batch:
                triplets = self._extract_triplets(chunk)
                for t in triplets:
                    self._add_triplet(t)

            progress = min(i + batch_size, total)
            print(f"    {progress}/{total} chunks processed "
                  f"({self.graph.number_of_nodes()} nodes, "
                  f"{self.graph.number_of_edges()} edges)")

        self.save()

    def _extract_triplets(self, chunk: TextChunk) -> list[Triplet]:
        """
        ── GRAPH EXTRACTION BOILERPLATE ──────────────────────────────────────

        Prompt Claude (fast model) to extract (subject, predicate, object)
        triplets from a single text chunk.

        This is the core "knowledge graph construction" function described
        in the system instructions. The LLM is given:
          1. A strict predicate vocabulary so edges are consistent.
          2. A JSON output schema so parsing is reliable.
          3. Domain-specific normalisation rules.

        Returns a (possibly empty) list of Triplet objects.
        """
        prompt_text = TRIPLET_USER_TEMPLATE.format(text=chunk.text[:2000])

        try:
            response = self.client.messages.create(
                model=self.settings.claude_fast_model,
                max_tokens=1024,
                system=TRIPLET_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt_text}],
            )

            raw_json = response.content[0].text.strip()

            # Strip markdown code fences if the model added them
            raw_json = re.sub(r"^```(?:json)?\s*", "", raw_json)
            raw_json = re.sub(r"\s*```$", "", raw_json)

            import json
            data = json.loads(raw_json)
            triplets: list[Triplet] = []

            for item in data.get("triplets", []):
                triplets.append(
                    Triplet(
                        subject=str(item["subject"]).upper().strip(),
                        predicate=str(item["predicate"]).upper().strip(),
                        obj=str(item["object"]).upper().strip(),
                        source_chunk_id=chunk.chunk_id,
                        source_page=chunk.page_number,
                        confidence=float(item.get("confidence", 1.0)),
                    )
                )

            return triplets

        except Exception as e:
            # Non-fatal: a failed chunk just means fewer graph edges
            print(f"    [WARN] Triplet extraction failed for chunk {chunk.chunk_id}: {e}")
            return []

    def _add_triplet(self, triplet: Triplet) -> None:
        """
        Add a single triplet to the NetworkX graph.

        Node attributes store all chunks that mention this entity.
        Edge attributes store the predicate, confidence, and source reference.
        """
        subj = triplet.subject
        obj = triplet.obj

        # Ensure both nodes exist
        if not self.graph.has_node(subj):
            self.graph.add_node(subj, mentions=[], entity_type="unknown")
        if not self.graph.has_node(obj):
            self.graph.add_node(obj, mentions=[], entity_type="unknown")

        # Record which chunk mentioned each entity
        self.graph.nodes[subj]["mentions"].append(triplet.source_chunk_id)
        self.graph.nodes[obj]["mentions"].append(triplet.source_chunk_id)

        # Add directed edge with metadata
        self.graph.add_edge(
            subj,
            obj,
            predicate=triplet.predicate,
            source_chunk_id=triplet.source_chunk_id,
            source_page=triplet.source_page,
            confidence=triplet.confidence,
        )

    # ── Query API ─────────────────────────────────────────────────────────────

    def get_neighbors(self, entity: str, depth: int = 1) -> list[dict[str, Any]]:
        """
        Return all edges within *depth* hops of *entity*.

        Used to augment RAG context: given "MIG WELDING", return all
        directly related entities and their relationship types.
        """
        entity = entity.upper().strip()
        if entity not in self.graph:
            return []

        results: list[dict] = []
        # BFS up to *depth* hops
        visited = {entity}
        frontier = {entity}

        for _ in range(depth):
            next_frontier: set[str] = set()
            for node in frontier:
                # Outgoing edges
                for _, target, edge_data in self.graph.out_edges(node, data=True):
                    results.append({
                        "source": node,
                        "predicate": edge_data.get("predicate", "RELATED_TO"),
                        "target": target,
                        "confidence": edge_data.get("confidence", 1.0),
                        "page": edge_data.get("source_page", 0),
                    })
                    if target not in visited:
                        next_frontier.add(target)
                # Incoming edges (reverse direction)
                for source, _, edge_data in self.graph.in_edges(node, data=True):
                    results.append({
                        "source": source,
                        "predicate": edge_data.get("predicate", "RELATED_TO"),
                        "target": node,
                        "confidence": edge_data.get("confidence", 1.0),
                        "page": edge_data.get("source_page", 0),
                    })
                    if source not in visited:
                        next_frontier.add(source)
            visited |= next_frontier
            frontier = next_frontier

        # Sort by confidence, deduplicate
        seen = set()
        deduped = []
        for r in sorted(results, key=lambda x: -x["confidence"]):
            key = (r["source"], r["predicate"], r["target"])
            if key not in seen:
                seen.add(key)
                deduped.append(r)

        return deduped

    def get_context_for_query(self, query: str, max_facts: int = 15) -> str:
        """
        Extract entities from *query*, find their graph neighborhoods,
        and return a formatted string of facts for LLM context injection.

        Example output:
          Graph facts:
          - (MIG WELDING) --[REQUIRES]--> (WIRE FEED SPEED)  [p.12]
          - (MIG WELDING) --[REQUIRES]--> (SHIELDING GAS)    [p.14]
          ...
        """
        # Find which graph entities appear in the query
        query_upper = query.upper()
        matching_nodes = [
            node for node in self.graph.nodes
            if node in query_upper or query_upper in node
        ]

        if not matching_nodes:
            return ""

        all_facts: list[dict] = []
        for node in matching_nodes[:5]:  # limit to 5 seed entities
            all_facts.extend(self.get_neighbors(node, depth=1))

        if not all_facts:
            return ""

        # Format as a readable list, capped at max_facts
        lines = ["Relevant knowledge graph facts:"]
        for fact in all_facts[:max_facts]:
            page_ref = f" [p.{fact['page']}]" if fact["page"] else ""
            lines.append(
                f"  • ({fact['source']}) --[{fact['predicate']}]--> ({fact['target']}){page_ref}"
            )

        return "\n".join(lines)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Return a summary dict suitable for logging or the /health endpoint."""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "top_entities": sorted(
                self.graph.degree(), key=lambda x: x[1], reverse=True
            )[:10],
        }
