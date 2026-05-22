import base64
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import numpy as np
import pdfplumber
from PIL import Image

logger = logging.getLogger(__name__)

# lazy singleton — downloads ~100MB on first call
_ocr_reader = None


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        logger.info("Loading EasyOCR model (first run may download ~100MB)...")
        _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        logger.info("EasyOCR ready.")
    return _ocr_reader


def _ocr_pil(img: Image.Image) -> str:
    try:
        reader = _get_ocr_reader()
        results = reader.readtext(np.array(img), detail=0, paragraph=True)
        return " ".join(results)
    except Exception as e:
        logger.warning(f"OCR failed: {e}")
        return ""


def _vision_ocr(img: Image.Image) -> str:
    """GPT-4o Vision — understands charts, math, diagrams. Falls back to EasyOCR if no key."""
    from config.settings import settings
    if not settings.openai_api_key:
        return _ocr_pil(img)
    try:
        from openai import OpenAI
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        resp = OpenAI(api_key=settings.openai_api_key).chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract ALL content from this page accurately:\n"
                            "- Text: transcribe exactly as written\n"
                            "- Charts/graphs: describe data, trends, axis labels, and key values\n"
                            "- Math formulas: write them out clearly\n"
                            "- Tables: preserve structure with | separators\n"
                            "Be thorough and precise."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
                    },
                ],
            }],
            max_tokens=2000,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"GPT-4o Vision failed, falling back to EasyOCR: {e}")
        return _ocr_pil(img)


@dataclass
class PageContent:
    page_number: int
    text: str
    tables: List[List[List[str]]] = field(default_factory=list)
    images_text: List[str] = field(default_factory=list)
    is_scanned: bool = False


def _is_scanned(page: fitz.Page, text: str) -> bool:
    return len(text.strip()) < 50 and len(page.get_images()) > 0


def _render_page_as_pil(page: fitz.Page) -> Image.Image:
    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat)
    return Image.open(io.BytesIO(pix.tobytes("png")))


def _extract_embedded_images(page: fitz.Page, doc: fitz.Document) -> List[str]:
    results = []
    for img_ref in page.get_images():
        try:
            xref = img_ref[0]
            base = doc.extract_image(xref)
            img = Image.open(io.BytesIO(base["image"]))
            if img.size[0] > 100 and img.size[1] > 100:
                # use Vision for embedded images — catches charts and diagrams
                text = _vision_ocr(img).strip()
                if text:
                    results.append(text)
        except Exception as e:
            logger.debug(f"Embedded image skipped: {e}")
    return results


def parse_pdf(pdf_path: Path) -> List[PageContent]:
    pdf_path = Path(pdf_path)
    pages: List[PageContent] = []

    fitz_doc = fitz.open(str(pdf_path))

    with pdfplumber.open(str(pdf_path)) as plumber_doc:
        for i in range(len(fitz_doc)):
            fitz_page = fitz_doc[i]
            plumber_page = plumber_doc.pages[i]
            page_num = i + 1

            text = fitz_page.get_text("text")
            scanned = _is_scanned(fitz_page, text)

            if scanned:
                logger.info(f"Page {page_num}: scanned — applying Vision OCR")
                text = _vision_ocr(_render_page_as_pil(fitz_page))

            tables: List = []
            try:
                for tbl in plumber_page.extract_tables():
                    if tbl:
                        tables.append(tbl)
            except Exception as e:
                logger.debug(f"Table extract error page {page_num}: {e}")

            images_text = [] if scanned else _extract_embedded_images(fitz_page, fitz_doc)

            table_blocks = []
            for tbl in tables:
                rows = [" | ".join(cell or "" for cell in row) for row in tbl if row]
                table_blocks.append("\n".join(rows))

            full_text = text
            if table_blocks:
                full_text += "\n[TABLE]\n" + "\n[TABLE]\n".join(table_blocks)
            if images_text:
                full_text += "\n[IMAGE_TEXT]\n" + "\n".join(images_text)

            pages.append(PageContent(
                page_number=page_num,
                text=full_text.strip(),
                tables=tables,
                images_text=images_text,
                is_scanned=scanned,
            ))

    fitz_doc.close()
    logger.info(f"Parsed {len(pages)} pages from '{pdf_path.name}'")
    return pages
