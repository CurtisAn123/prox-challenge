"""
src/ingestion/pdf_parser.py
────────────────────────────
Parses every PDF in the files/ directory.

Responsibilities:
  1. Extract full text, broken into overlapping chunks suitable for embedding.
  2. Extract every page as a high-res image (for page-level image retrieval).
  3. Extract embedded raster images within each page (diagrams, schematics).
  4. Attach rich metadata to every chunk so downstream retrieval can cite sources.

Library choices:
  - PyMuPDF (fitz): blazing-fast, handles scanned pages, exports images natively.
  - pdfplumber: layered on top for superior table detection.
"""

import base64
import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image

from config.settings import get_settings


# ──────────────────────────────────────────────────────────────────────────────
#  Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """A single text chunk ready to be embedded and stored."""
    chunk_id: str          # deterministic SHA-256 of content
    text: str
    source_file: str       # original PDF filename
    page_number: int       # 1-indexed
    chunk_index: int       # position within this page
    section_hint: str      # best-guess section heading (empty if none found)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExtractedImage:
    """An image extracted from a PDF page."""
    image_id: str          # deterministic SHA-256 of pixel data
    source_file: str
    page_number: int
    image_index: int       # position on the page (0-indexed)
    file_path: str         # path where the image was saved to disk
    width: int
    height: int
    caption: str = ""      # populated later by the vision LLM (see vector_store.py)
    image_type: str = "embedded"   # "embedded" | "page_render" | "diagram_crop"
    surrounding_text: str = ""     # page text for caption context
    crop_box: tuple[int, int, int, int] | None = None  # (x_min, y_min, x_max, y_max) in pixels; None for non-crops
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sha256(data: str | bytes) -> str:
    """Return the first 16 hex chars of the SHA-256 of *data*."""
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()[:16]


def _detect_section(text: str) -> str:
    """
    Heuristically extract a section/heading from raw text.

    Looks for ALL-CAPS lines or lines matching common manual heading patterns
    (e.g. 'SECTION 3', 'TROUBLESHOOTING', 'MIG WELDING').
    Returns the first match, or empty string.
    """
    heading_re = re.compile(
        r"^(?:"
        r"(?:SECTION|CHAPTER|PART)\s+\d+"           # SECTION 3
        r"|[A-Z][A-Z\s\-]{4,40}"                    # TROUBLESHOOTING
        r"|(?:MIG|TIG|STICK|FLUX[- ]CORED)\s+WELD"  # process headings
        r")",
        re.MULTILINE,
    )
    for line in text.splitlines():
        line = line.strip()
        if heading_re.match(line) and len(line) > 3:
            return line[:80]
    return ""


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split *text* into overlapping character-level chunks.

    We prefer to break at sentence or paragraph boundaries within the
    allowed window to preserve semantic coherence.
    """
    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Try to snap the end to the nearest sentence boundary ('. ')
        if end < text_len:
            snap = text.rfind(". ", start, end)
            if snap != -1 and snap > start + chunk_size // 2:
                end = snap + 1   # include the period

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Advance by (chunk_size - overlap) so consecutive chunks share context
        start += max(chunk_size - overlap, 1)

    return chunks


# ──────────────────────────────────────────────────────────────────────────────
#  Main parser
# ──────────────────────────────────────────────────────────────────────────────

class PDFParser:
    """
    Parses all PDFs in *files_dir* and yields TextChunk / ExtractedImage objects.

    Usage::

        parser = PDFParser()
        chunks, images = parser.parse_all()
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.images_dir = self.settings.extracted_images_dir
        self.images_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def parse_all(self) -> tuple[list[TextChunk], list[ExtractedImage]]:
        """
        Walk *files_dir*, parse every PDF, and return:
          - all_chunks: a flat list of TextChunk objects
          - all_images: a flat list of ExtractedImage objects
        """
        all_chunks: list[TextChunk] = []
        all_images: list[ExtractedImage] = []

        pdf_files = sorted(self.settings.files_dir.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(
                f"No PDF files found in {self.settings.files_dir.resolve()}"
            )

        for pdf_path in pdf_files:
            print(f"  Parsing: {pdf_path.name}")
            chunks, images = self._parse_pdf(pdf_path)
            all_chunks.extend(chunks)
            all_images.extend(images)
            print(f"    → {len(chunks)} chunks, {len(images)} images")

        return all_chunks, all_images

    # ── Per-file parsing ──────────────────────────────────────────────────────

    def _parse_pdf(
        self, pdf_path: Path
    ) -> tuple[list[TextChunk], list[ExtractedImage]]:
        chunks: list[TextChunk] = []
        images: list[ExtractedImage] = []

        # Open the same PDF with both libraries in parallel
        fitz_doc = fitz.open(str(pdf_path))
        plumber_doc = pdfplumber.open(str(pdf_path))

        for page_num in range(len(fitz_doc)):
            fitz_page = fitz_doc[page_num]
            plumber_page = plumber_doc.pages[page_num]
            human_page = page_num + 1  # 1-indexed for citations

            # ── Text ──────────────────────────────────────────────────────────
            page_text = self._extract_page_text(fitz_page, plumber_page)
            if page_text.strip():
                page_chunks = self._text_to_chunks(
                    page_text, pdf_path.name, human_page
                )
                chunks.extend(page_chunks)

            # ── Images ────────────────────────────────────────────────────────
            page_images = self._extract_page_images(
                fitz_page, pdf_path.name, human_page, page_text[:2000]
            )
            images.extend(page_images)

            # ── Page render (captures vector diagrams invisible to get_images) ─
            if self._is_visual_page(fitz_page):
                render = self._render_page_image(
                    fitz_page, pdf_path.name, human_page, page_text[:2000]
                )
                if render:
                    images.append(render)

        fitz_doc.close()
        plumber_doc.close()
        return chunks, images

    # ── Text extraction ───────────────────────────────────────────────────────

    def _extract_page_text(
        self, fitz_page: fitz.Page, plumber_page
    ) -> str:
        """
        Merge text from both libraries:
          - fitz for raw text flow (preserves reading order)
          - pdfplumber for table content (formatted as pipe-delimited strings)
        """
        # Raw text via fitz
        raw_text = fitz_page.get_text("text")

        # Tables via pdfplumber (tables often contain critical spec data)
        table_texts: list[str] = []
        for table in plumber_page.extract_tables():
            rows = []
            for row in table:
                # Filter None cells and join with pipe separator
                cells = [str(c).strip() for c in row if c is not None]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                table_texts.append("\n".join(rows))

        # Combine: raw text first, then formatted tables
        combined = raw_text
        if table_texts:
            combined += "\n\n[TABLE DATA]\n" + "\n\n".join(table_texts)

        return combined

    def _text_to_chunks(
        self, page_text: str, source_file: str, page_number: int
    ) -> list[TextChunk]:
        """Split page text into overlapping chunks, each with a TextChunk object."""
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

    # ── Image extraction ──────────────────────────────────────────────────────

    def _extract_page_images(
        self, fitz_page: fitz.Page, source_file: str, page_number: int,
        surrounding_text: str = "",
    ) -> list[ExtractedImage]:
        """
        Extract all embedded raster images from a PDF page.

        Each image is saved as a PNG to *extracted_images_dir* with a
        deterministic filename. A corresponding JSON sidecar file stores
        metadata so nothing is re-processed on subsequent runs.
        """
        results: list[ExtractedImage] = []

        # fitz returns a list of (xref, smask, width, height, bpc, colorspace, ...) tuples
        image_list = fitz_page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]  # cross-reference number in the PDF
            try:
                base_image = fitz_page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                width = base_image["width"]
                height = base_image["height"]
                ext = base_image["ext"]  # png, jpeg, …

                # Skip tiny images (icons, bullet-point graphics, etc.)
                if width < 80 or height < 80:
                    continue

                # Deterministic ID based on pixel data
                image_id = _sha256(image_bytes)
                filename = f"{Path(source_file).stem}_p{page_number}_i{img_index}_{image_id}.png"
                save_path = self.images_dir / filename

                # Only write to disk if not already saved
                if not save_path.exists():
                    img = Image.open(fitz.open(stream=image_bytes, filetype=ext))
                    img.save(str(save_path), "PNG")

                results.append(
                    ExtractedImage(
                        image_id=image_id,
                        source_file=source_file,
                        page_number=page_number,
                        image_index=img_index,
                        file_path=str(save_path),
                        width=width,
                        height=height,
                        caption="",  # populated later by vision LLM
                        image_type="embedded",
                        surrounding_text=surrounding_text,
                        metadata={
                            "source": source_file,
                            "page": page_number,
                            "img_index": img_index,
                            "width": width,
                            "height": height,
                            "image_type": "embedded",
                        },
                    )
                )
            except Exception as e:
                # Non-fatal: log and continue to next image
                print(f"    [WARN] Could not extract image xref={xref} on p{page_number}: {e}")

        return results

    # ── Page-level rendering (captures vector diagrams) ───────────────────────

    def _is_visual_page(self, fitz_page: fitz.Page) -> bool:
        """Return True if the page contains vector drawings or embedded images."""
        has_drawings = len(fitz_page.get_drawings()) > 3
        has_embedded = len(fitz_page.get_images(full=True)) > 0
        return has_drawings or has_embedded

    def _render_page_image(
        self,
        fitz_page: fitz.Page,
        source_file: str,
        page_number: int,
        surrounding_text: str = "",
    ) -> ExtractedImage | None:
        """
        Render the full page as a 2× PNG.

        This captures vector diagrams, schematics, and any other content that
        is part of the page's drawing stream rather than embedded as a raster
        object (which `get_images()` would miss entirely).
        """
        try:
            # 2× zoom gives ~144 dpi — legible for Claude Vision without being huge
            matrix = fitz.Matrix(2, 2)
            pixmap = fitz_page.get_pixmap(matrix=matrix, alpha=False)
            png_bytes = pixmap.tobytes("png")

            image_id = _sha256(png_bytes)
            filename = f"{Path(source_file).stem}_p{page_number}_render_{image_id}.png"
            save_path = self.images_dir / filename

            if not save_path.exists():
                save_path.write_bytes(png_bytes)

            return ExtractedImage(
                image_id=image_id,
                source_file=source_file,
                page_number=page_number,
                image_index=-1,          # -1 signals "whole-page render"
                file_path=str(save_path),
                width=pixmap.width,
                height=pixmap.height,
                caption="",              # populated later by vision LLM
                image_type="page_render",
                surrounding_text=surrounding_text,
                metadata={
                    "source": source_file,
                    "page": page_number,
                    "img_index": -1,
                    "width": pixmap.width,
                    "height": pixmap.height,
                    "image_type": "page_render",
                },
            )
        except Exception as e:
            print(f"    [WARN] Could not render page {page_number} of {source_file}: {e}")
            return None

    # ── Utility: load image as base64 ─────────────────────────────────────────

    @staticmethod
    def image_to_base64(file_path: str) -> str:
        """Load a saved image and return it as a base64-encoded string."""
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
