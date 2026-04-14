"""
src/agent/strategic_planner.py
────────────────────────────────
Lightweight query decomposition using Claude Haiku.

The StrategicPlanner converts a raw user query into an ordered list of
sub-tasks (a Plan) that the ReasoningLoop uses as a starting guide for its
first Think step.  The plan is non-binding — the loop adapts freely based
on observations — but it reduces wasted tool calls on straightforward queries
by giving the model an immediate sense of the best execution order.

Failure is always non-fatal: if Haiku returns invalid JSON or times out,
a single ``final_answer`` sub-task is returned so the loop can still run.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import anthropic

from src.agent.prompts import build_prompt, PLANNER_SYSTEM_TEMPLATE


# ──────────────────────────────────────────────────────────────────────────────
#  Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SubTask:
    """A single step in the agent's execution plan."""
    task_type: str    # entity_lookup | relationship_traversal | text_retrieval |
                      # image_retrieval | tool_call | final_answer
    description: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    """An ordered list of sub-tasks produced by the StrategicPlanner."""
    sub_tasks: list[SubTask] = field(default_factory=list)

    def as_preamble(self) -> str:
        """
        Serialise the plan as plain text to prepend to the first user message
        in the ReasoningLoop conversation.
        """
        if not self.sub_tasks:
            return ""

        lines = ["Suggested plan (adapt based on observations):"]
        for i, task in enumerate(self.sub_tasks, 1):
            param_hint = ""
            if task.params:
                # Show the first param value as a compact hint
                first_val = next(iter(task.params.values()), "")
                param_hint = f" — {first_val}" if first_val else ""
            lines.append(f"{i}. [{task.task_type}] {task.description}{param_hint}")

        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
#  StrategicPlanner
# ──────────────────────────────────────────────────────────────────────────────

_FALLBACK_PLAN = Plan(sub_tasks=[SubTask(task_type="final_answer", description="Answer directly")])


class StrategicPlanner:
    """
    Decomposes a user query into an ordered Plan using Claude Haiku.

    Parameters
    ----------
    client : anthropic.Anthropic
        Shared Anthropic client.
    fast_model : str
        The Haiku model ID (from settings.claude_fast_model).
    product_name : str
        Substituted into the planner system prompt (from settings.product_name).
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        fast_model: str,
        product_name: str,
    ) -> None:
        self.client = client
        self.fast_model = fast_model
        self._system_prompt = build_prompt(PLANNER_SYSTEM_TEMPLATE, product_name)

    def plan(self, query: str) -> Plan:
        """
        Decompose *query* into a Plan.

        Returns the fallback single-step plan on any error so the
        ReasoningLoop can always proceed.
        """
        try:
            response = self.client.messages.create(
                model=self.fast_model,
                max_tokens=512,
                system=self._system_prompt,
                messages=[{"role": "user", "content": query}],
            )

            raw = response.content[0].text.strip()
            # Strip markdown fences if the model wrapped the JSON
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            data = json.loads(raw)
            sub_tasks = [
                SubTask(
                    task_type=str(item.get("task_type", "text_retrieval")),
                    description=str(item.get("description", "")),
                    params=dict(item.get("params", {})),
                )
                for item in data.get("sub_tasks", [])
            ]

            if not sub_tasks:
                return _FALLBACK_PLAN

            return Plan(sub_tasks=sub_tasks)

        except Exception as exc:
            print(f"[WARN] StrategicPlanner failed ({type(exc).__name__}: {exc}); using fallback plan.")
            return _FALLBACK_PLAN
