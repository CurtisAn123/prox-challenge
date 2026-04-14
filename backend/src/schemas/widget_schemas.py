"""
src/schemas/widget_schemas.py
───────────────────────────────
Interactive widget schemas for parameter-driven queries.

When a user asks "what wire speed for 1/4 inch steel with MIG?" the text
answer requires them to mentally cross-reference a table. Instead, we
return a structured JSON schema that the frontend renders as an interactive
configurator widget.

Each schema follows this shape:
    {
        "component": str,       # React component name the frontend renders
        "title": str,
        "description": str,
        "fields": [...],        # input fields with labels, types, options
        "output_fields": [...], # computed output fields
        "pre_populated": {...}, # values extracted from the query
        "source_pages": [...]   # populated at runtime by vector store retrieval
    }

Widget registry (active):
    wire_speed        — recommend wire speed + voltage for material/thickness
    troubleshooting   — interactive decision tree for fault diagnosis
    process_selector  — choose welding process for material/application

Source pages and dropdown options are populated at runtime by querying the
vector store and knowledge graph — nothing is hardcoded.
"""

from __future__ import annotations

import copy
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.ingestion.vector_store import VectorStore
    from src.agent.kg_interface import KGInterface


# ──────────────────────────────────────────────────────────────────────────────
#  Base widget schemas (UI structure only — no data values, no page numbers)
# ──────────────────────────────────────────────────────────────────────────────

_WIRE_SPEED_BASE: dict[str, Any] = {
    "component": "WireSpeedConfigurator",
    "title": "Wire Speed & Voltage Configurator",
    "description": (
        "Get recommended wire feed speed and voltage settings for "
        "MIG or Flux-Cored welding based on your material and thickness."
    ),
    "fields": [
        {
            "name": "process",
            "label": "Welding Process",
            "type": "select",
            "options": ["MIG (ER70S-6)", "Flux-Cored (E71T-GS)"],
            "required": True,
        },
        {
            "name": "material",
            "label": "Base Material",
            "type": "select",
            "options": ["Mild Steel", "Stainless Steel", "Aluminum"],
            "required": True,
        },
        {
            "name": "thickness",
            "label": "Material Thickness",
            "type": "select",
            "options": [
                '24 gauge (0.024")',
                '22 gauge (0.030")',
                '3/16"',
                '1/4"',
                '5/16"',
                '3/8"',
            ],
            "required": True,
        },
        {
            "name": "position",
            "label": "Weld Position",
            "type": "select",
            "options": ["Flat", "Horizontal", "Vertical", "Overhead"],
            "required": False,
            "default": "Flat",
        },
    ],
    "output_fields": [
        {"name": "wire_feed_speed", "label": "Wire Feed Speed",     "unit": "IPM"},
        {"name": "voltage",         "label": "Voltage",             "unit": "V"},
        {"name": "wire_diameter",   "label": "Recommended Wire Diameter", "unit": "inch"},
        {"name": "shielding_gas",   "label": "Shielding Gas"},
        {"name": "tip_to_work",     "label": "Tip-to-Work Distance", "unit": "inch"},
    ],
    "pre_populated": {},
    "source_pages": [],
    "notes": (
        "Settings are starting points from the OmniPro 220 manual synergic "
        "chart. Fine-tune by listening for the characteristic sizzling sound."
    ),
}


_TROUBLESHOOTING_BASE: dict[str, Any] = {
    "component": "TroubleshootingFlowchart",
    "title": "Weld Fault Troubleshooter",
    "description": (
        "Interactive troubleshooting guide for common weld defects. "
        "Answer each question to identify the root cause and fix."
    ),
    "fields": [
        {
            "name": "symptom",
            "label": "What problem are you seeing?",
            "type": "select",
            "options": [
                "Porosity (holes/pits in weld)",
                "Spatter (excessive splatter)",
                "Incomplete fusion / cold lap",
                "Undercut",
                "Wire burn-back / bird-nesting",
                "Arc won't start",
                "Weld too wide / shallow",
                "Weld too narrow / high crown",
                "Excessive smoke",
                "Machine overheating",
            ],
            "required": True,
        },
        {
            "name": "process",
            "label": "Welding process",
            "type": "select",
            "options": ["MIG", "Flux-Cored", "TIG", "Stick"],
            "required": True,
        },
    ],
    "output_fields": [
        {"name": "likely_causes", "label": "Likely Causes",          "type": "list"},
        {"name": "steps",         "label": "Troubleshooting Steps",  "type": "ordered_list"},
        {"name": "prevention",    "label": "Prevention Tips",        "type": "list"},
    ],
    "pre_populated": {},
    "source_pages": [],
}


_PROCESS_SELECTOR_BASE: dict[str, Any] = {
    "component": "ProcessSelector",
    "title": "Welding Process Selector",
    "description": (
        "Not sure which process to use? Answer a few questions "
        "and we'll recommend the right mode on the OmniPro 220."
    ),
    "fields": [
        {
            "name": "material",
            "label": "What are you welding?",
            "type": "select",
            "options": [
                "Mild steel (structural)",
                "Thin sheet metal (auto body)",
                "Stainless steel",
                "Aluminum",
                "Cast iron",
            ],
            "required": True,
        },
        {
            "name": "environment",
            "label": "Where are you welding?",
            "type": "select",
            "options": ["Indoors (controlled)", "Outdoors / windy"],
            "required": True,
        },
        {
            "name": "skill_level",
            "label": "Your experience level",
            "type": "select",
            "options": ["Beginner", "Intermediate", "Advanced"],
            "required": True,
        },
    ],
    "output_fields": [
        {"name": "recommended_process", "label": "Recommended Process"},
        {"name": "reason",              "label": "Why this process"},
        {"name": "initial_settings",    "label": "Starting Settings"},
    ],
    "pre_populated": {},
    "source_pages": [],
}


# ──────────────────────────────────────────────────────────────────────────────
#  Retrieval queries — used to resolve source pages from the vector store
# ──────────────────────────────────────────────────────────────────────────────

_SOURCE_QUERIES: dict[str, str] = {
    "wire_speed":       "wire feed speed voltage settings chart synergic table thickness",
    "troubleshooting":  "weld defect troubleshooting causes solutions porosity spatter",
    "process_selector": "welding process selection guide material application MIG TIG Stick",
}

# KG predicates to query per widget for option enrichment
_KG_ENRICH: dict[str, list[tuple[str, str | None, str]]] = {
    # (predicate, subject_filter, field_name_to_enrich)
    "wire_speed":       [("COMPATIBLE_WITH", None, "material")],
    "troubleshooting":  [("FIXED_BY",  None, "symptom"),
                         ("CAUSED_BY", None, "symptom")],
    "process_selector": [("COMPATIBLE_WITH", None, "material"),
                         ("USED_FOR", None, "material")],
}


# ──────────────────────────────────────────────────────────────────────────────
#  Registry
# ──────────────────────────────────────────────────────────────────────────────

class WidgetSchemaRegistry:
    """
    Looks up the appropriate widget schema for a given query.

    Source pages and dropdown options are populated at runtime by querying
    the vector store and knowledge graph — no data values are hardcoded.

    Parameters
    ----------
    vector_store  : VectorStore   — used to resolve source_pages at call time
    kg_interface  : KGInterface   — used to supplement dropdown options
    """

    _bases: dict[str, dict[str, Any]] = {
        "wire_speed":       _WIRE_SPEED_BASE,
        "troubleshooting":  _TROUBLESHOOTING_BASE,
        "process_selector": _PROCESS_SELECTOR_BASE,
    }

    # Keyword-to-widget mapping (fast path before LLM extraction)
    _keyword_map: list[tuple[list[str], str]] = [
        (["wire speed", "wire feed", "voltage for", "settings for", "inch steel",
          "gauge", "thickness", "mm steel"], "wire_speed"),
        (["porosity", "spatter", "undercut", "cold lap", "bird nest",
          "burn back", "not fusing", "defect", "problem with my weld",
          "troubleshoot", "why is my weld"], "troubleshooting"),
        (["which process", "what process", "should i use mig", "should i use tig",
          "best process", "recommend a process"], "process_selector"),
    ]

    def __init__(
        self,
        vector_store: "VectorStore",
        kg_interface: "KGInterface",
    ) -> None:
        self.vs = vector_store
        self.kg = kg_interface

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_widget_by_key(
        self,
        widget_key: str,
        pre_populated: dict | None = None,
    ) -> dict[str, Any]:
        """
        Direct key lookup — used by AgentRouter when the LLM has already
        selected the widget via the return_widget tool.

        Source pages are resolved from the vector store; dropdown options
        are enriched from the knowledge graph.
        Falls back to wire_speed if widget_key is not recognised.
        """
        key = widget_key if widget_key in self._bases else "wire_speed"
        schema = self._build_schema(key, pre_populated)
        return {"type": "widget", "component": schema["component"], "schema": schema}

    def get_widget(
        self,
        query: str,
        pre_populated: dict | None = None,
    ) -> dict[str, Any]:
        """
        Return the best-matching widget schema for *query*.

        Keyword-matches the query to a widget key, then resolves live
        source pages and KG-enriched options.
        """
        query_lower = query.lower()
        widget_key = "wire_speed"
        for keywords, key in self._keyword_map:
            if any(kw in query_lower for kw in keywords):
                widget_key = key
                break

        schema = self._build_schema(widget_key, pre_populated)
        return {"type": "widget", "component": schema["component"], "schema": schema}

    # ── Internals ──────────────────────────────────────────────────────────────

    def _build_schema(
        self,
        widget_key: str,
        pre_populated: dict | None = None,
    ) -> dict[str, Any]:
        """Deep-copy the base schema, then inject live source pages and KG options."""
        schema = copy.deepcopy(self._bases[widget_key])

        # 1. Resolve source pages from the vector store
        schema["source_pages"] = self._get_source_pages(widget_key)

        # 2. Supplement dropdown options from the KG
        self._enrich_options(schema, widget_key)

        # 3. Inject pre-populated values if provided
        if pre_populated:
            schema["pre_populated"] = pre_populated

        return schema

    def _get_source_pages(self, widget_key: str) -> list[int]:
        """
        Query the vector store with the widget's topic query and return the
        page numbers of the top matching chunks.
        """
        query = _SOURCE_QUERIES.get(widget_key, "")
        if not query:
            return []
        try:
            results = self.vs.search_text(query, top_k=4)
            return sorted({r["page"] for r in results if r.get("page")})
        except Exception:
            return []

    def _enrich_options(self, schema: dict[str, Any], widget_key: str) -> None:
        """
        Query the KG for relevant entities and append any new values to the
        matching field's options list. Existing defaults are kept first;
        KG-derived entries are appended only if not already represented.
        Mutates *schema* in place.
        """
        enrichments = _KG_ENRICH.get(widget_key, [])
        if not enrichments:
            return

        # Build a lookup of field name → field dict
        fields_by_name = {f["name"]: f for f in schema.get("fields", [])}

        for predicate, subject_filter, field_name in enrichments:
            field = fields_by_name.get(field_name)
            if field is None or field.get("type") != "select":
                continue

            try:
                result = self.kg.query_by_predicate(predicate, subject=subject_filter)
                kg_entities: set[str] = set()

                for item in result.get("results", []):
                    # FIXED_BY / CAUSED_BY: the fault/symptom is the subject
                    # COMPATIBLE_WITH / USED_FOR: the material/process is the target
                    if predicate in ("FIXED_BY", "CAUSED_BY"):
                        kg_entities.add(item["from"])
                    else:
                        kg_entities.add(item["to"])

                # Merge: keep existing options first, append new KG entities
                existing = field["options"]
                field["options"] = _merge_options(existing, kg_entities)

            except Exception:
                pass  # KG unavailable — defaults remain unchanged


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _merge_options(existing: list[str], kg_entities: set[str]) -> list[str]:
    """
    Append KG entity names to *existing* options, skipping any that are
    already represented (case-insensitive substring match).

    KG entities are uppercase (e.g. "MILD STEEL") — title-cased before
    appending (e.g. "Mild Steel").
    """
    result = list(existing)
    existing_lower = " ".join(o.lower() for o in existing)

    for entity in sorted(kg_entities):
        formatted = entity.title()
        # Skip if the entity name already appears anywhere in existing options
        if formatted.lower() not in existing_lower:
            result.append(formatted)

    return result
