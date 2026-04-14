"""
src/agent/kg_interface.py
──────────────────────────
Product-agnostic interface layer over the raw KnowledgeGraph.

Provides three query modes the ReasoningLoop can call as tools:

  search_entity(name)             — fuzzy-match an entity and return its
                                    1-hop neighborhood from the KG.
  find_path(from_entity, to)      — shortest path between two entities,
                                    returned as an ordered list of hops.
  query_by_predicate(pred, subj?) — filter all edges by predicate type,
                                    optionally anchored to a subject.

The fuzzy matching layer (difflib + substring fallback) compensates for the
KG's normalised uppercase entity names without requiring the caller to know
the exact stored form of every entity.
"""

from __future__ import annotations

import difflib
from typing import Any

import networkx as nx

from src.ingestion.graph_builder import KnowledgeGraph


class KGInterface:
    """
    High-level, tool-friendly interface to the KnowledgeGraph.

    All methods return plain dicts with an ``error`` key set to a string
    when something goes wrong, so the LLM can read the error and adapt
    without raising exceptions that would break the tool loop.
    """

    def __init__(self, knowledge_graph: KnowledgeGraph) -> None:
        self.kg = knowledge_graph

    # ── Public tool methods ────────────────────────────────────────────────────

    def search_entity(self, entity_name: str) -> dict[str, Any]:
        """
        Fuzzy-match *entity_name* against graph nodes and return its
        1-hop neighborhood.

        Matching strategy (in order):
          1. Exact match (case-normalised to uppercase).
          2. difflib.get_close_matches with cutoff=0.6.
          3. Substring containment (both directions).

        Returns
        -------
        dict with keys:
            matched_entity : str | None  — the node name that was found
            match_score    : float       — 1.0 for exact, 0.0–1.0 for fuzzy
            neighbors      : list[dict]  — from kg.get_neighbors(depth=1)
            error          : str | None
        """
        matched, score = self._fuzzy_match_node(entity_name)

        if matched is None:
            return {
                "matched_entity": None,
                "match_score": 0.0,
                "neighbors": [],
                "error": (
                    f"No entity matching '{entity_name}' found in the knowledge graph. "
                    f"Try a broader term or use search_text instead."
                ),
            }

        neighbors = self.kg.get_neighbors(matched, depth=1)
        return {
            "matched_entity": matched,
            "match_score": round(score, 3),
            "neighbors": neighbors,
            "error": None,
        }

    def find_path(self, from_entity: str, to_entity: str) -> dict[str, Any]:
        """
        Find the shortest relationship path between two entities.

        Uses an undirected projection of the MultiDiGraph so that edges
        can be traversed in either direction — the path still reports the
        original directed predicates for each hop.

        Returns
        -------
        dict with keys:
            path   : list[dict]  — [{from, predicate, to, page}, ...]
            length : int
            error  : str | None
        """
        from_matched, _ = self._fuzzy_match_node(from_entity)
        to_matched, _   = self._fuzzy_match_node(to_entity)

        if from_matched is None:
            return {
                "path": [], "length": 0,
                "error": f"Could not find entity '{from_entity}' in the knowledge graph.",
            }
        if to_matched is None:
            return {
                "path": [], "length": 0,
                "error": f"Could not find entity '{to_entity}' in the knowledge graph.",
            }
        if from_matched == to_matched:
            return {"path": [], "length": 0, "error": "Source and target are the same entity."}

        # Project to undirected for path-finding; then recover edge predicates
        undirected = nx.Graph(self.kg.graph)

        try:
            node_path: list[str] = nx.shortest_path(undirected, from_matched, to_matched)
        except nx.NetworkXNoPath:
            return {
                "path": [], "length": 0,
                "error": (
                    f"No path found between '{from_matched}' and '{to_matched}' "
                    f"in the knowledge graph."
                ),
            }
        except nx.NodeNotFound as exc:
            return {"path": [], "length": 0, "error": str(exc)}

        hops: list[dict] = []
        for i in range(len(node_path) - 1):
            u, v = node_path[i], node_path[i + 1]
            predicate, page = self._best_edge_predicate(u, v)
            hops.append({"from": u, "predicate": predicate, "to": v, "page": page})

        return {"path": hops, "length": len(hops), "error": None}

    def query_by_predicate(
        self,
        predicate: str,
        subject: str | None = None,
    ) -> dict[str, Any]:
        """
        Return all edges in the graph that have a given predicate type.

        Parameters
        ----------
        predicate : str
            One of the KG vocabulary predicates: PART_OF, CONNECTED_TO,
            REQUIRES, CONTROLS, USED_FOR, FIXED_BY, CAUSED_BY,
            COMPATIBLE_WITH, DEPENDS_ON, LOCATED_AT, RANGE_IS,
            DEFAULT_IS, WARNING. Case-insensitive.
        subject : str | None
            If provided, further filter to edges whose source node
            fuzzy-matches this name.

        Returns
        -------
        dict with keys:
            results : list[dict]  — [{from, predicate, to, page}, ...]
            count   : int
            error   : str | None
        """
        pred_upper = predicate.upper().strip()
        results: list[dict] = []

        subj_matched: str | None = None
        if subject:
            subj_matched, _ = self._fuzzy_match_node(subject)
            if subj_matched is None:
                return {
                    "results": [],
                    "count": 0,
                    "error": f"Could not find subject entity '{subject}' in the knowledge graph.",
                }

        for u, v, edge_data in self.kg.graph.edges(data=True):
            edge_pred = edge_data.get("predicate", "").upper()
            if edge_pred != pred_upper:
                continue
            if subj_matched and u != subj_matched:
                continue
            results.append({
                "from": u,
                "predicate": edge_pred,
                "to": v,
                "page": edge_data.get("source_page", 0),
            })

        return {"results": results, "count": len(results), "error": None}

    # ── Internals ──────────────────────────────────────────────────────────────

    def _fuzzy_match_node(
        self, name: str, cutoff: float = 0.6
    ) -> tuple[str | None, float]:
        """
        Return (best_node, score) for *name* against the graph's node list.

        Score is 1.0 for exact match, the difflib ratio for fuzzy match,
        or 0.5 for a substring containment fallback.
        Returns (None, 0.0) if no match passes the cutoff.
        """
        if not self.kg.graph.number_of_nodes():
            return None, 0.0

        normalised = name.upper().strip()
        nodes: list[str] = list(self.kg.graph.nodes)

        # 1. Exact match
        if normalised in self.kg.graph:
            return normalised, 1.0

        # 2. difflib close match
        close = difflib.get_close_matches(normalised, nodes, n=1, cutoff=cutoff)
        if close:
            score = difflib.SequenceMatcher(None, normalised, close[0]).ratio()
            return close[0], score

        # 3. Substring containment (both directions)
        for node in nodes:
            if normalised in node or node in normalised:
                return node, 0.5

        return None, 0.0

    def _best_edge_predicate(self, u: str, v: str) -> tuple[str, int]:
        """
        For a pair of nodes (u, v), return the predicate and page of the
        highest-confidence edge between them (in either direction).
        """
        best_pred = "RELATED_TO"
        best_page = 0
        best_conf = -1.0

        # Check directed u→v
        if self.kg.graph.has_edge(u, v):
            for _, edge_data in self.kg.graph[u][v].items():
                conf = edge_data.get("confidence", 0.0)
                if conf > best_conf:
                    best_conf = conf
                    best_pred = edge_data.get("predicate", "RELATED_TO")
                    best_page = edge_data.get("source_page", 0)

        # Check reverse v→u (undirected traversal may pick this direction)
        if self.kg.graph.has_edge(v, u) and best_conf < 0:
            for _, edge_data in self.kg.graph[v][u].items():
                conf = edge_data.get("confidence", 0.0)
                if conf > best_conf:
                    best_conf = conf
                    best_pred = edge_data.get("predicate", "RELATED_TO")
                    best_page = edge_data.get("source_page", 0)

        return best_pred, best_page
