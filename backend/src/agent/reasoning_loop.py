"""
src/agent/reasoning_loop.py
────────────────────────────
ReAct (Reason + Act) loop — the core reasoning engine.

Each iteration:
  Think  — Claude reads the conversation history (including all prior
            tool observations) and decides what to do next.
  Act    — Claude emits one or more tool_use blocks.
  Observe — The loop executes each tool, appends the results as tool_result
             messages, and feeds them back to Claude.

The loop terminates when:
  - Claude calls the ``finish`` tool (normal completion).
  - Claude produces an end_turn with plain text and no tool call (treated
    as a text answer — shouldn't happen with good prompting, but handled
    gracefully).
  - MAX_ITERATIONS is reached (safety cap, returns a polite fallback message).

Source accumulation:
  Every tool that returns text chunks, KG facts, or images contributes
  citations to the accumulated_sources list, which is attached to the
  final LoopResult for the frontend's citation panel.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import anthropic

from src.agent.multimodal_manager import MultimodalManager
from src.agent.strategic_planner import Plan
from src.agent.tool_executor import ToolExecutor, REASONING_TOOLS


# ──────────────────────────────────────────────────────────────────────────────
#  Result dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LoopResult:
    """Typed result returned by ReasoningLoop.run()."""
    answer: str
    response_type: str                    # "text" | "mermaid" | "image" | "rich" | "widget"
    sources: list[dict[str, Any]]
    mermaid_syntax: str | None = None
    images: list[dict[str, Any]] | None = None
    intent_label: str = "reasoning"
    widget_data: dict[str, Any] | None = None  # populated when response_type == "widget"


# ──────────────────────────────────────────────────────────────────────────────
#  ReasoningLoop
# ──────────────────────────────────────────────────────────────────────────────

class ReasoningLoop:
    """
    Executes the ReAct Think/Act/Observe cycle until the model calls finish().

    Parameters
    ----------
    anthropic_client  : anthropic.Anthropic
    reasoning_model   : str          — Opus model ID for deep reasoning
    tool_executor     : ToolExecutor — dispatches tool_use blocks
    multimodal_manager: MultimodalManager — kept as a reference (not called directly)
    product_name      : str          — for logging; logic is product-agnostic
    system_prompt     : str          — fully rendered REASONING_SYSTEM_TEMPLATE
    """

    MAX_ITERATIONS = 8

    def __init__(
        self,
        anthropic_client: anthropic.Anthropic,
        reasoning_model: str,
        tool_executor: ToolExecutor,
        multimodal_manager: MultimodalManager,
        product_name: str,
        system_prompt: str,
    ) -> None:
        self.client = anthropic_client
        self.model = reasoning_model
        self.executor = tool_executor
        self.mm = multimodal_manager
        self.product_name = product_name
        self.system_prompt = system_prompt

    # ── Public entry point ────────────────────────────────────────────────────

    def run(
        self,
        query: str,
        plan: Plan,
        status_callback: Callable[[str, dict], None] | None = None,
    ) -> LoopResult:
        """
        Execute the ReAct loop for *query*, guided by the strategic *plan*.

        Returns a LoopResult regardless of how the loop terminates.

        *status_callback*, if provided, is called at each Think and tool-call
        step so callers can stream progress to the frontend.  Signature::

            status_callback(event_type: str, data: dict) -> None

        Event types emitted: ``"thinking"``, ``"tool_call"``.
        """
        # Build the initial user message: plan preamble + query
        preamble = plan.as_preamble()
        first_content = (
            f"{preamble}\n\nUser question: {query}"
            if preamble else
            f"User question: {query}"
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": first_content}
        ]

        accumulated_sources: list[dict[str, Any]] = []
        accumulated_images: list[dict[str, Any]] = []

        for iteration in range(self.MAX_ITERATIONS):
            # ── Think ─────────────────────────────────────────────────────────
            if status_callback:
                status_callback("thinking", {})
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self.system_prompt,
                tools=REASONING_TOOLS,
                messages=messages,
            )

            # Append the full assistant turn verbatim so the context is intact
            messages.append({"role": "assistant", "content": response.content})

            # ── end_turn without a tool call — treat as final text answer ─────
            if response.stop_reason == "end_turn":
                text = _extract_text(response.content)
                return LoopResult(
                    answer=text or "(No answer generated.)",
                    response_type="text",
                    sources=accumulated_sources,
                    intent_label="reasoning",
                )

            if response.stop_reason != "tool_use":
                # Unexpected stop reason — bail out safely
                break

            # ── Act: collect tool_use blocks ──────────────────────────────────
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            tool_result_contents: list[dict[str, Any]] = []
            finish_result: LoopResult | None = None

            for block in tool_use_blocks:
                if status_callback:
                    status_callback("tool_call", {"tool": block.name})

                # ── finish() terminates the loop ──────────────────────────────
                if block.name == "finish":
                    inp: dict[str, Any] = block.input
                    resp_type = inp.get("type", "text")
                    finish_images = inp.get("images") or accumulated_images or None
                    # "image" and "rich" both carry images; all other types suppress them
                    should_carry_images = resp_type in ("image", "rich")
                    finish_result = LoopResult(
                        answer=inp.get("answer", ""),
                        response_type=resp_type,
                        sources=accumulated_sources,
                        mermaid_syntax=inp.get("mermaid_syntax"),
                        images=finish_images if should_carry_images else None,
                        intent_label="reasoning",
                    )
                    # Still need to append a tool_result so the API contract
                    # is satisfied (finish tool_use must have a result)
                    tool_result_contents.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps({"ok": True}),
                    })
                    continue  # process remaining blocks; return after the loop

                # ── return_widget() terminates the loop with a widget response ─
                elif block.name == "return_widget":
                    inp = block.input
                    finish_result = LoopResult(
                        answer=inp.get("summary", ""),
                        response_type="widget",
                        sources=accumulated_sources,
                        widget_data={
                            "widget_key": inp.get("widget_key", "wire_speed"),
                            "pre_populated": inp.get("pre_populated"),
                        },
                        intent_label="reasoning",
                    )
                    tool_result_contents.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps({"ok": True}),
                    })
                    continue

                # ── Observe: execute the tool ─────────────────────────────────
                raw_result = self.executor.execute(block.name, block.input)

                # Accumulate citations and images (must happen before scrubbing)
                self._accumulate_sources(block.name, raw_result, accumulated_sources)
                self._accumulate_images(block.name, raw_result, accumulated_images)

                # Strip base64 payloads before serialising into the message
                # history — images are already captured in accumulated_images.
                serialisable = _scrub_base64(raw_result, block.name)
                tool_result_contents.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(serialisable),
                })

            # If finish() was called, return now (after all blocks processed)
            if finish_result is not None:
                return finish_result

            # Feed all observations back as a single user turn
            messages.append({"role": "user", "content": tool_result_contents})

        # ── Iteration cap exceeded ────────────────────────────────────────────
        return LoopResult(
            answer=(
                "I was unable to fully answer your question within the reasoning limit. "
                "Please try rephrasing or breaking the question into smaller parts."
            ),
            response_type="text",
            sources=accumulated_sources,
            intent_label="reasoning_timeout",
        )

    # ── Internals ──────────────────────────────────────────────────────────────

    def _accumulate_sources(
        self,
        tool_name: str,
        result: dict[str, Any],
        sources: list[dict[str, Any]],
    ) -> None:
        """Extract citation metadata from tool results and add to sources."""
        if not result.get("ok"):
            return

        data = result.get("result", {})

        if tool_name == "search_text":
            for chunk in data.get("chunks", []):
                sources.append({
                    "source": chunk.get("source", ""),
                    "page":   chunk.get("page", 0),
                    "section": chunk.get("section", ""),
                    "score":  chunk.get("score", 0.0),
                })

        elif tool_name in ("search_kg_entity", "find_kg_path", "query_by_predicate"):
            # KG results carry page numbers on edges
            neighbors = data.get("neighbors", data.get("path", data.get("results", [])))
            for item in neighbors:
                page = item.get("page", 0)
                if page:
                    sources.append({
                        "source": "Knowledge Graph",
                        "page": page,
                        "section": "",
                        "score": item.get("confidence", 1.0),
                    })

        elif tool_name in ("retrieve_image", "analyze_image_with_context"):
            images = (
                data.get("images", [])
                if tool_name == "retrieve_image"
                else ([data.get("image_data")] if data.get("image_data") else [])
            )
            for img in images:
                if img:
                    sources.append({
                        "source": img.get("source", ""),
                        "page":   img.get("page", 0),
                        "caption": img.get("caption", ""),
                        "score":  img.get("score", 0.0),
                    })

        elif tool_name == "generate_diagram":
            for src in data.get("context_sources", []):
                sources.append(src)

        elif tool_name == "calculate_duty_cycle":
            for page in data.get("source_pages", []):
                sources.append({
                    "source": f"{self.product_name} Owner's Manual",
                    "page": page,
                    "section": "Duty Cycle",
                    "score": 1.0,
                })

    def _accumulate_images(
        self,
        tool_name: str,
        result: dict[str, Any],
        images: list[dict[str, Any]],
    ) -> None:
        """Stash image dicts so finish(type='image') can reference them."""
        if not result.get("ok"):
            return

        data = result.get("result", {})

        if tool_name == "retrieve_image":
            images.extend(data.get("images", []))
        elif tool_name == "analyze_image_with_context":
            img = data.get("image_data")
            if img:
                images.append(img)

        # Deduplicate by file_path, then cap at 2.
        MAX_IMAGES = 2
        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for img in images:
            fp = img.get("file_path") or img.get("caption", str(img))
            if fp not in seen:
                seen.add(fp)
                deduped.append(img)
        images[:] = deduped[:MAX_IMAGES]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_text(content: list[Any]) -> str:
    """Pull the first text block from an Anthropic message content list."""
    for block in content:
        if hasattr(block, "text"):
            return block.text.strip()
        if isinstance(block, dict) and block.get("type") == "text":
            return block.get("text", "").strip()
    return ""


def _scrub_base64(result: dict[str, Any], tool_name: str) -> dict[str, Any]:
    """
    Return a copy of *result* with base64_data stripped from image payloads.

    Images are already captured in accumulated_images before this is called,
    so the conversation history only needs the lightweight metadata
    (caption, page, score) — not the raw PNG bytes.
    """
    _IMAGE_TOOLS = {"retrieve_image", "analyze_image_with_context"}
    if tool_name not in _IMAGE_TOOLS or not result.get("ok"):
        return result

    import copy
    scrubbed = copy.deepcopy(result)
    data = scrubbed.get("result", {})

    for img in data.get("images", []):
        img.pop("base64_data", None)

    img_data = data.get("image_data")
    if isinstance(img_data, dict):
        img_data.pop("base64_data", None)

    return scrubbed
