"""
src/tools/duty_cycle_tool.py
─────────────────────────────
Agentic tool: Duty Cycle Lookup for the Vulcan OmniPro 220.

Provides:
  - DUTY_CYCLE_TOOL_SCHEMA  — Anthropic tool definition (JSON schema) that
                              describes the function to Claude.

The tool is implemented as a retrieval tool in ToolExecutor._calculate_duty_cycle,
which queries the knowledge graph (KG) and vector store for duty cycle
specifications directly from the manual. No values are hardcoded here.
"""

from __future__ import annotations

from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
#  Tool schema (Anthropic tool-use API format)
# ──────────────────────────────────────────────────────────────────────────────

DUTY_CYCLE_TOOL_SCHEMA: dict[str, Any] = {
    "name": "calculate_duty_cycle",
    "description": (
        "Retrieve duty cycle specifications from the Vulcan OmniPro 220 manual "
        "and knowledge graph for a given welding process, supply voltage, and "
        "output current. Returns relevant manual excerpts and KG facts — read "
        "the manual_excerpts and kg_range_facts in the result to determine the "
        "actual duty cycle percentage, on-time, rest time, and any thermal "
        "warnings documented in the manual. "
        "Call this tool ONLY when the user explicitly asks about duty cycle, "
        "how long they can weld continuously, overheating risk, thermal limits, "
        "or required cool-down time. "
        "Do NOT call this for questions about welding settings, wire speed, "
        "voltage parameters, or process setup — use search_text for those."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "process": {
                "type": "string",
                "enum": ["MIG", "Flux-Cored", "TIG", "Stick"],
                "description": (
                    "The welding process being used. Must be one of the four "
                    "modes supported by the OmniPro 220: 'MIG' (wire-feed with "
                    "shielding gas), 'Flux-Cored' (self-shielded or gas-shielded "
                    "FCAW), 'TIG' (tungsten inert-gas, lift-arc or HF-start), "
                    "or 'Stick' (SMAW)."
                ),
            },
            "input_voltage": {
                "type": "string",
                "enum": ["120V", "240V"],
                "description": (
                    "The AC supply voltage the machine is connected to. "
                    "'120V' is a standard US household single-phase outlet. "
                    "'240V' is a dedicated single-phase 220–240 V circuit. "
                    "When the user has not specified, ask."
                ),
            },
            "output_amps": {
                "type": "integer",
                "minimum": 20,
                "maximum": 220,
                "description": (
                    "The output current in whole amperes at which the user "
                    "intends to weld. Valid range: 20 A to 220 A. "
                    "If the user gave a decimal (e.g. 187.5 A), round to the "
                    "nearest integer before calling this tool."
                ),
            },
        },
        "required": ["process", "input_voltage", "output_amps"],
    },
}
