"""
src/agent/multimodal_manager.py
────────────────────────────────
Handles Vision-Language Model (VLM) analysis of retrieved images.

The ReasoningLoop calls this when the agent decides — based on what it reads
in retrieved text passages — that a visual element needs deeper analysis.
The LLM explicitly requests analysis via the analyze_image_with_context tool;
there is no automatic figure-reference scanning.

Workflow for analyze():
  1. Use ImageRetriever to find the image that best matches the figure reference.
  2. If a match is found (score > threshold), send the base64-encoded image
     together with the surrounding text context to the VLM.
  3. Return the VLM's narrative analysis plus the raw ImageResult dict so
     the ReasoningLoop can accumulate it for the final response.
"""

from __future__ import annotations

from typing import Any

import anthropic

from src.retrieval.image_retriever import ImageRetriever


_VLM_SYSTEM = """\
You are a technical image analyst. You will be shown an image from a product
manual alongside a passage of text that references it.

Your job:
1. Describe what the image shows in precise technical terms.
2. Identify all labelled components, connectors, settings, or measurements.
3. Explain how the image relates to the text passage provided.
4. Note anything in the image that adds information not present in the text.

Be concise and factual. Do not speculate beyond what is visible.
"""

_IMAGE_SIMILARITY_THRESHOLD = 0.25


class MultimodalManager:
    """
    Retrieves images and analyses them with a Vision-Language Model.

    Parameters
    ----------
    image_retriever : ImageRetriever
        Retrieves matching images from the vector store by semantic caption search.
    anthropic_client : anthropic.Anthropic
        Shared Anthropic client (created once at startup in main.py).
    vision_model : str
        Model ID to use for VLM analysis (from settings.claude_vision_model).
    """

    def __init__(
        self,
        image_retriever: ImageRetriever,
        anthropic_client: anthropic.Anthropic,
        vision_model: str,
    ) -> None:
        self.image_retriever = image_retriever
        self.client = anthropic_client
        self.vision_model = vision_model

    def analyze(self, figure_ref: str, context_text: str) -> dict[str, Any]:
        """
        Retrieve the image best matching *figure_ref* and analyse it with the VLM.

        Parameters
        ----------
        figure_ref : str
            The figure reference or visual description to search for
            (e.g. "front panel controls", "wire feed mechanism").
        context_text : str
            The surrounding text passage that mentions the visual element.
            Passed to the VLM as grounding context.

        Returns
        -------
        dict with keys:
            figure_ref  : str        — echoed back for traceability
            analysis    : str        — VLM narrative (empty string on failure)
            image_data  : dict|None  — ImageResult.to_dict() or None
            error       : str|None
        """
        # ── Step 1: Retrieve the best matching image ──────────────────────────
        try:
            images = self.image_retriever.retrieve(figure_ref)
        except Exception as exc:
            return {
                "figure_ref": figure_ref,
                "analysis": "",
                "image_data": None,
                "error": f"Image retrieval failed: {exc}",
            }

        if not images:
            return {
                "figure_ref": figure_ref,
                "analysis": "",
                "image_data": None,
                "error": f"No images found matching '{figure_ref}'.",
            }

        top = images[0]
        if top.score < _IMAGE_SIMILARITY_THRESHOLD:
            return {
                "figure_ref": figure_ref,
                "analysis": "",
                "image_data": None,
                "error": (
                    f"Best image match for '{figure_ref}' had a low similarity score "
                    f"({top.score:.2f}). No reliable image found."
                ),
            }

        # ── Step 2: Send to VLM ───────────────────────────────────────────────
        try:
            response = self.client.messages.create(
                model=self.vision_model,
                max_tokens=512,
                system=_VLM_SYSTEM,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": top.base64_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    f"Text passage that references this image:\n\n"
                                    f"{context_text}\n\n"
                                    f"Analyse the image in relation to this passage."
                                ),
                            },
                        ],
                    }
                ],
            )
            analysis = response.content[0].text.strip()
        except Exception as exc:
            return {
                "figure_ref": figure_ref,
                "analysis": "",
                "image_data": top.to_dict(),
                "error": f"VLM analysis failed: {exc}",
            }

        return {
            "figure_ref": figure_ref,
            "analysis": analysis,
            "image_data": top.to_dict(),
            "error": None,
        }
