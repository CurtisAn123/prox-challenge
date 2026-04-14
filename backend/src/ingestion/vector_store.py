"""
src/ingestion/vector_store.py
──────────────────────────────
ChromaDB-backed vector store with two collections:

  text_chunks  — embedded text passages from the PDFs
  image_index  — embedded image captions (so images are searchable by text)

Why ChromaDB?
  - Embedded in-process: no Docker, no server, runs on any laptop.
  - Persistent on disk: survives restarts without re-ingestion.
  - Upgrade path: the interface here maps directly to Qdrant if you need
    a production-grade ANN index later.

Embedding strategy:
  - sentence-transformers/all-MiniLM-L6-v2 (local, fast, no API key).
  - For images, we caption them with Claude Vision and embed the caption.
    This means image search is purely text-driven — no CLIP needed.
"""

import base64
import hashlib
import io
import json
from pathlib import Path
from typing import Any

from PIL import Image

import anthropic
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from config.settings import get_settings
from src.ingestion.pdf_parser import ExtractedImage, TextChunk, PDFParser


# ──────────────────────────────────────────────────────────────────────────────
#  Image captioning prompt
# ──────────────────────────────────────────────────────────────────────────────

IMAGE_CAPTION_SYSTEM = """\
You are an expert at describing technical diagrams and images from welding equipment manuals.
Describe this image concisely but completely, covering:
- What type of image it is (diagram, photo, schematic, chart, table, etc.)
- What components, controls, labels, or values are visible
- What technical concept it illustrates
- How a user would use this image (e.g. "to identify the correct polarity setting")
- For diagram crops: use the surrounding page text and the label (if provided in brackets
  at the start of the context) to explain what this diagram illustrates and how a user
  would apply it (e.g. "Use this to identify the correct polarity wiring for TIG welding").

Keep your description under 200 words. Be specific — include exact labels, numbers, and
component names visible in the image.
"""

DIAGRAM_DETECTION_SYSTEM = """\
You are an expert at analysing technical manual pages. Identify all distinct diagrams,
figures, schematics, wiring diagrams, charts, or labelled illustrations on the page.

Do NOT include: running body text paragraphs, page headers, footers, page numbers, or
decorative borders/rules.

Return ONLY valid JSON — no markdown fences, no explanation, nothing else:
{
  "diagrams": [
    {"label": "<short description of what this diagram shows>", "x_min": <int>, "y_min": <int>, "x_max": <int>, "y_max": <int>}
  ]
}

Coordinates are pixel values relative to the top-left corner of the image (origin 0,0).
If there are no distinct diagrams on this page, return: {"diagrams": []}
"""


# ──────────────────────────────────────────────────────────────────────────────
#  VectorStore
# ──────────────────────────────────────────────────────────────────────────────

class VectorStore:
    """
    Manages two ChromaDB collections:
      - ``text_chunks``: text from PDFs, embedded for semantic retrieval.
      - ``image_index``: image captions, enabling text → image retrieval.
    """

    COLLECTION_TEXT = "text_chunks"
    COLLECTION_IMAGE = "image_index"

    def __init__(self) -> None:
        self.settings = get_settings()
        self.anthropic_client = anthropic.Anthropic(
            api_key=self.settings.anthropic_api_key
        )

        # Initialise local embedding model
        print(f"  Loading embedding model: {self.settings.embedding_model}")
        self.embedder = SentenceTransformer(self.settings.embedding_model)

        # Initialise ChromaDB (persistent on disk)
        chroma_path = Path(self.settings.chroma_db_path)
        chroma_path.mkdir(parents=True, exist_ok=True)
        self.chroma = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Get or create collections
        self.text_collection = self.chroma.get_or_create_collection(
            name=self.COLLECTION_TEXT,
            metadata={"hnsw:space": "cosine"},
        )
        self.image_collection = self.chroma.get_or_create_collection(
            name=self.COLLECTION_IMAGE,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def is_populated(self) -> bool:
        """Return True if the text collection already has documents."""
        return self.text_collection.count() > 0

    def add_text_chunks(self, chunks: list[TextChunk]) -> None:
        """
        Embed and store all text chunks.

        Chunks are upserted (safe to run multiple times — duplicates are
        detected by chunk_id and overwritten rather than duplicated).
        """
        print(f"  Embedding {len(chunks)} text chunks…")

        # Process in batches to avoid hitting memory limits
        batch_size = 64
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            texts = [c.text for c in batch]
            ids = [c.chunk_id for c in batch]
            metadatas = [
                {
                    "source": c.source_file,
                    "page": c.page_number,
                    "section": c.section_hint,
                    "chunk_index": c.chunk_index,
                }
                for c in batch
            ]

            # Compute embeddings locally
            embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()

            self.text_collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            print(f"    Embedded {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

    def add_images(self, images: list[ExtractedImage]) -> None:
        """
        Caption each image with Claude Vision, then embed the caption.

        For page_render images, diagram detection runs first. If Claude Vision
        finds distinct diagram regions, those crops are indexed instead of the
        full page. If detection finds nothing (text-only page or API failure),
        the full-page render is indexed as a fallback.

        For embedded images and diagram_crop images, captioning proceeds directly.
        """
        print(f"  Captioning and indexing {len(images)} images…")

        # Expand page_renders into diagram crops where detection succeeds.
        to_index: list[ExtractedImage] = []
        for img in images:
            if img.image_type == "page_render":
                crops = self._detect_diagram_regions(img)
                if crops:
                    to_index.extend(crops)
                else:
                    to_index.append(img)
            else:
                to_index.append(img)

        # Caption and upsert each image.
        for img in to_index:
            # Skip if already indexed
            existing = self.image_collection.get(ids=[img.image_id])
            if existing["ids"]:
                continue

            # Generate caption via Claude Vision
            caption = self._caption_image(img)
            if not caption:
                continue

            img.caption = caption
            embedding = self.embedder.encode([caption], show_progress_bar=False)[0].tolist()

            self.image_collection.upsert(
                ids=[img.image_id],
                documents=[caption],
                embeddings=[embedding],
                metadatas=[
                    {
                        "source": img.source_file,
                        "page": img.page_number,
                        "file_path": img.file_path,
                        "width": img.width,
                        "height": img.height,
                        "caption": caption,
                        "image_type": img.image_type,
                        "surrounding_text": img.surrounding_text[:400],
                        "crop_label": img.metadata.get("crop_label", ""),
                    }
                ],
            )
            print(f"    Indexed image: {Path(img.file_path).name} (p.{img.page_number})")

    # ── Query API ─────────────────────────────────────────────────────────────

    def search_text(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """
        Semantic search over text chunks.

        Returns a list of result dicts, each containing:
          - text: the chunk content
          - score: cosine similarity (0→1, higher is better)
          - metadata: source, page, section, etc.
        """
        k = top_k or self.settings.top_k_text
        query_embedding = self.embedder.encode([query], show_progress_bar=False)[0].tolist()

        results = self.text_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.text_collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )

        output: list[dict] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append(
                {
                    "text": doc,
                    "score": 1.0 - dist,  # convert distance to similarity
                    "source": meta.get("source", "unknown"),
                    "page": meta.get("page", 0),
                    "section": meta.get("section", ""),
                }
            )

        return output

    def search_images(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """
        Semantic search over image captions.

        Returns result dicts with image metadata and file_path for
        the ImageRetriever to load and return.
        """
        k = top_k or self.settings.top_k_images
        if self.image_collection.count() == 0:
            return []

        query_embedding = self.embedder.encode([query], show_progress_bar=False)[0].tolist()

        results = self.image_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.image_collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        output: list[dict] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append(
                {
                    "caption": doc,
                    "score": 1.0 - dist,
                    "source": meta.get("source", "unknown"),
                    "page": meta.get("page", 0),
                    "file_path": meta.get("file_path", ""),
                    "width": meta.get("width", 0),
                    "height": meta.get("height", 0),
                }
            )

        return output

    # ── Image captioning (private) ────────────────────────────────────────────

    def _caption_image(self, img: ExtractedImage) -> str:
        """
        Use Claude Vision to generate a descriptive caption for an image.

        Returns the caption string, or empty string on failure.
        """
        try:
            with open(img.file_path, "rb") as f:
                image_data = base64.standard_b64encode(f.read()).decode("utf-8")

            context_line = (
                f"\nContext from the same page:\n{img.surrounding_text}\n"
                if img.surrounding_text else ""
            )
            response = self.anthropic_client.messages.create(
                model=self.settings.claude_vision_model,
                max_tokens=300,
                system=IMAGE_CAPTION_SYSTEM,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    f"This image is from page {img.page_number} of "
                                    f"'{img.source_file}'.{context_line}"
                                    f"Please describe it."
                                ),
                            },
                        ],
                    }
                ],
            )

            return response.content[0].text.strip()

        except Exception as e:
            print(f"    [WARN] Caption failed for {img.file_path}: {e}")
            return ""

    def _detect_diagram_regions(
        self,
        img: ExtractedImage,
    ) -> list[ExtractedImage]:
        """
        Use Claude Vision to detect distinct diagram regions on a full-page render.

        Returns a list of new ExtractedImage objects with image_type="diagram_crop".
        Returns an empty list if detection fails or no diagrams are found — the
        caller (add_images) will then fall back to indexing the full page.

        Each crop PNG is saved to self.settings.extracted_images_dir with the naming:
            {stem}_p{page}_crop_{x_min}_{y_min}_{x_max}_{y_max}_{sha256}.png
        """
        # ── Load full-page PNG bytes ──────────────────────────────────────────
        try:
            with open(img.file_path, "rb") as f:
                raw_bytes = f.read()
            image_data = base64.standard_b64encode(raw_bytes).decode("utf-8")
        except OSError as e:
            print(f"    [WARN] Could not read {img.file_path} for diagram detection: {e}")
            return []

        # ── Ask Claude Vision to locate diagram bounding boxes ────────────────
        try:
            response = self.anthropic_client.messages.create(
                model=self.settings.claude_vision_model,
                max_tokens=1024,
                system=DIAGRAM_DETECTION_SYSTEM,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    f"This is page {img.page_number} of '{img.source_file}'.\n\n"
                                    f"Page text:\n{img.surrounding_text}\n\n"
                                    f"Identify all distinct diagrams or figures and return "
                                    f"their bounding boxes as pixel coordinates."
                                ),
                            },
                        ],
                    }
                ],
            )
            raw_text = response.content[0].text.strip()
        except Exception as e:
            print(f"    [WARN] Diagram detection API call failed for p{img.page_number}: {e}")
            return []

        # ── Parse the JSON response ───────────────────────────────────────────
        try:
            # Strip accidental markdown fences Claude might prepend
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
            detection = json.loads(raw_text)
            diagrams = detection.get("diagrams", [])
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"    [WARN] Malformed JSON from diagram detection on p{img.page_number}: {e}")
            return []

        if not diagrams:
            return []

        # ── Open the full-page PNG once for all crops ─────────────────────────
        try:
            full_image = Image.open(io.BytesIO(raw_bytes))
            img_w, img_h = full_image.size
        except Exception as e:
            print(f"    [WARN] PIL could not open {img.file_path}: {e}")
            return []

        crops: list[ExtractedImage] = []
        images_dir = Path(self.settings.extracted_images_dir)
        images_dir.mkdir(parents=True, exist_ok=True)

        for i, diag in enumerate(diagrams):
            # ── Validate and clamp bounding box ──────────────────────────────
            try:
                x_min = max(0, int(diag["x_min"]))
                y_min = max(0, int(diag["y_min"]))
                x_max = min(img_w, int(diag["x_max"]))
                y_max = min(img_h, int(diag["y_max"]))
            except (KeyError, TypeError, ValueError) as e:
                print(f"    [WARN] Invalid bbox for diagram {i} on p{img.page_number}: {e}")
                continue

            box_w = x_max - x_min
            box_h = y_max - y_min
            MIN_DIM = 50
            if box_w < MIN_DIM or box_h < MIN_DIM:
                print(f"    [WARN] Skipping tiny box ({box_w}×{box_h}) on p{img.page_number}")
                continue

            # ── Crop and save ─────────────────────────────────────────────────
            crop = full_image.crop((x_min, y_min, x_max, y_max))
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            crop_bytes = buf.getvalue()

            crop_id = hashlib.sha256(crop_bytes).hexdigest()[:16]
            stem = Path(img.source_file).stem
            filename = (
                f"{stem}_p{img.page_number}"
                f"_crop_{x_min}_{y_min}_{x_max}_{y_max}_{crop_id}.png"
            )
            save_path = images_dir / filename

            if not save_path.exists():
                save_path.write_bytes(crop_bytes)

            label = diag.get("label", f"diagram {i + 1}")
            # Prepend the crop label so the captioner always knows what this diagram
            # is for, even if the page text doesn't open with that topic.
            crop_surrounding_text = f"[{label}] {img.surrounding_text}"

            crops.append(
                ExtractedImage(
                    image_id=crop_id,
                    source_file=img.source_file,
                    page_number=img.page_number,
                    image_index=i,
                    file_path=str(save_path),
                    width=box_w,
                    height=box_h,
                    caption="",
                    image_type="diagram_crop",
                    surrounding_text=crop_surrounding_text,
                    crop_box=(x_min, y_min, x_max, y_max),
                    metadata={
                        "source": img.source_file,
                        "page": img.page_number,
                        "image_type": "diagram_crop",
                        "crop_label": label,
                        "parent_image_id": img.image_id,
                        "width": box_w,
                        "height": box_h,
                    },
                )
            )
            print(f"    Detected diagram '{label}' on p{img.page_number} → {filename}")

        return crops
