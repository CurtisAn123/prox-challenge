"""
src/ingestion/mistral_ocr.py
─────────────────────────────
PDF parsing via Mistral OCR (mistral-ocr-latest).

Mistral OCR processes each PDF page and returns:
  - Structured markdown text (superior to layout-based extraction for scanned content)
  - Detected image regions as base64-encoded PNGs with bounding boxes

Extracted images are saved to disk and returned as ExtractedImage objects with
image_type="ocr_detected". The downstream VectorStore.add_images() passes them
directly to Claude for captioning — Mistral owns detection, Claude owns analysis.
"""

import base64
import io
import re
from pathlib import Path

from PIL import Image

from config.settings import get_settings
from src.ingestion.pdf_parser import (
    ExtractedImage,
    TextChunk,
    _chunk_text,
    _detect_section,
    _sha256,
)

# Regex to strip Mistral's image placeholder syntax: ![img-N](img-N)
_IMAGE_PLACEHOLDER_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


class MistralOCRParser:
    """
    Parses all PDFs in files_dir using Mistral OCR.

    Returns the same (list[TextChunk], list[ExtractedImage]) tuple as PDFParser
    so the rest of the ingestion pipeline requires no changes.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.images_dir = self.settings.extracted_images_dir
        self.images_dir.mkdir(parents=True, exist_ok=True)

        from mistralai import Mistral
        self._client = Mistral(api_key=self.settings.mistral_api_key)

    # ── Public API ─────────────────────────────────────────────────────────────

    def parse_all(self) -> tuple[list[TextChunk], list[ExtractedImage]]:
        """Walk files_dir, OCR every PDF, return all chunks and images."""
        all_chunks: list[TextChunk] = []
        all_images: list[ExtractedImage] = []

        pdf_files = sorted(self.settings.files_dir.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(
                f"No PDF files found in {self.settings.files_dir.resolve()}"
            )

        for pdf_path in pdf_files:
            print(f"  OCR processing: {pdf_path.name}")
            chunks, images = self._parse_pdf(pdf_path)
            all_chunks.extend(chunks)
            all_images.extend(images)
            print(f"    → {len(chunks)} chunks, {len(images)} images")

        return all_chunks, all_images

    # ── Per-file processing ────────────────────────────────────────────────────

    def _parse_pdf(
        self, pdf_path: Path
    ) -> tuple[list[TextChunk], list[ExtractedImage]]:
        chunks: list[TextChunk] = []
        images: list[ExtractedImage] = []

        # Upload the PDF to Mistral for OCR processing
        file_id = self._upload_pdf(pdf_path)
        try:
            ocr_response = self._run_ocr(file_id)
            for page in ocr_response.pages:
                human_page = page.index + 1  # 1-indexed for citations

                # ── Text ──────────────────────────────────────────────────────
                clean_text = _IMAGE_PLACEHOLDER_RE.sub("", page.markdown).strip()
                if clean_text:
                    page_chunks = self._text_to_chunks(
                        clean_text, pdf_path.name, human_page
                    )
                    chunks.extend(page_chunks)

                # ── Images ────────────────────────────────────────────────────
                page_images = self._extract_page_images(
                    page, pdf_path.name, human_page, clean_text[:2000]
                )
                images.extend(page_images)
        finally:
            self._cleanup(file_id)

        return chunks, images

    # ── Upload / OCR / cleanup ─────────────────────────────────────────────────

    def _upload_pdf(self, pdf_path: Path) -> str:
        """Upload the PDF to Mistral Files API and return the file_id."""
        with open(pdf_path, "rb") as f:
            uploaded = self._client.files.upload(
                file={"file_name": pdf_path.name, "content": f},
                purpose="ocr",
            )
        return uploaded.id

    def _run_ocr(self, file_id: str):
        """Get signed URL and run OCR, returning the full OCR response."""
        signed = self._client.files.get_signed_url(file_id=file_id)
        return self._client.ocr.process(
            model=self.settings.mistral_ocr_model,
            document={"type": "document_url", "document_url": signed.url},
            include_image_base64=True,
        )

    def _cleanup(self, file_id: str) -> None:
        """Delete the uploaded file from Mistral after processing."""
        try:
            self._client.files.delete(file_id=file_id)
        except Exception as e:
            print(f"    [WARN] Could not delete Mistral file {file_id}: {e}")

    # ── Text chunking ──────────────────────────────────────────────────────────

    def _text_to_chunks(
        self, page_text: str, source_file: str, page_number: int
    ) -> list[TextChunk]:
        raw_chunks = _chunk_text(
            page_text,
            self.settings.chunk_size,
            self.settings.chunk_overlap,
        )
        section = _detect_section(page_text)

        result: list[TextChunk] = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk_id = _sha256(f"{source_file}:p{page_number}:c{i}:{chunk_text[:50]}")
            result.append(
                TextChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source_file=source_file,
                    page_number=page_number,
                    chunk_index=i,
                    section_hint=section,
                    metadata={
                        "source": source_file,
                        "page": page_number,
                        "chunk_index": i,
                        "section": section,
                        "char_count": len(chunk_text),
                    },
                )
            )
        return result

    # ── Image extraction ───────────────────────────────────────────────────────

    def _extract_page_images(
        self, page, source_file: str, page_number: int, surrounding_text: str
    ) -> list[ExtractedImage]:
        """
        Save each Mistral-detected image to disk and return ExtractedImage objects.

        Mistral OCR returns images with:
          - image_base64: raw base64 bytes (no data URI prefix)
          - id: e.g. "img-0-p1"
          - top_left_x/y, bottom_right_x/y: normalized 0–1 bounding box coords
        """
        results: list[ExtractedImage] = []

        for img_obj in page.images or []:
            try:
                raw_b64 = img_obj.image_base64 or ""
                if not raw_b64:
                    continue

                # Strip data URI prefix if present (e.g. "data:image/png;base64,...")
                if "," in raw_b64:
                    raw_b64 = raw_b64.split(",", 1)[1]

                image_bytes = base64.b64decode(raw_b64)

                # Decode to get dimensions
                pil_img = Image.open(io.BytesIO(image_bytes))
                width, height = pil_img.size

                if width < 80 or height < 80:
                    continue

                image_id = _sha256(image_bytes)
                stem = Path(source_file).stem
                filename = f"{stem}_p{page_number}_ocr_{img_obj.id}_{image_id}.png"
                save_path = self.images_dir / filename

                if not save_path.exists():
                    # Re-save as PNG for consistency
                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    save_path.write_bytes(buf.getvalue())

                # Bounding box: normalize 0–1 coords → pixel coords
                crop_box = self._normalized_bbox(img_obj, width, height)

                results.append(
                    ExtractedImage(
                        image_id=image_id,
                        source_file=source_file,
                        page_number=page_number,
                        image_index=len(results),
                        file_path=str(save_path),
                        width=width,
                        height=height,
                        caption="",  # populated later by Claude Vision
                        image_type="ocr_detected",
                        surrounding_text=surrounding_text,
                        crop_box=crop_box,
                        metadata={
                            "source": source_file,
                            "page": page_number,
                            "img_index": len(results),
                            "width": width,
                            "height": height,
                            "image_type": "ocr_detected",
                            "ocr_image_id": img_obj.id,
                        },
                    )
                )
            except Exception as e:
                print(
                    f"    [WARN] Could not process OCR image {getattr(img_obj, 'id', '?')} "
                    f"on p{page_number}: {e}"
                )

        return results

    @staticmethod
    def _normalized_bbox(
        img_obj, width: int, height: int
    ) -> tuple[int, int, int, int] | None:
        """
        Convert Mistral's normalized (0–1) bounding box to pixel coordinates.
        Returns None if the attributes are missing or zero.
        """
        try:
            x_min = int(getattr(img_obj, "top_left_x", 0) * width)
            y_min = int(getattr(img_obj, "top_left_y", 0) * height)
            x_max = int(getattr(img_obj, "bottom_right_x", 1) * width)
            y_max = int(getattr(img_obj, "bottom_right_y", 1) * height)
            if x_max > x_min and y_max > y_min:
                return (x_min, y_min, x_max, y_max)
        except (TypeError, ValueError):
            pass
        return None
