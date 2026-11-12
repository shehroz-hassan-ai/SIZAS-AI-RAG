import re
from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.config import MAX_FILE_SIZE_MB


@dataclass
class PageText:
    page_number: int
    text: str


class PDFLoadError(Exception):
    """PDF load na ho sake to ye error aata hai."""


def clean_text(text: str) -> str:
    """Faltu formatting hatao."""
    text = text.replace("\x00", "")
    text = re.sub(r"-\n(\w)", r"\1", text)   # line ke end par kati hui word jodo
    text = re.sub(r"\n{3,}", "\n\n", text)   # bohat saari khali lines kam karo
    text = re.sub(r"[ \t]+", " ", text)      # extra spaces hatao
    return text.strip()


def load_pdf(file_bytes: bytes, filename: str) -> list[PageText]:
    """PDF bytes se page-wise text return karta hai."""
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise PDFLoadError(
            f"File too large ({size_mb:.1f} MB). Limit {MAX_FILE_SIZE_MB} MB."
        )

    try:
        reader = PdfReader(BytesIO(file_bytes))
        if reader.is_encrypted:
            raise PDFLoadError("PDF password protected hai.")
    except PdfReadError:
        raise PDFLoadError(f"'{filename}' valid PDF nahi hai.")
    except PDFLoadError:
        raise
    except Exception as e:
        raise PDFLoadError(f"PDF open nahi hui: {e}")

    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append(PageText(page_number=i, text=text))

    if not pages:
        raise PDFLoadError(
            "PDF se text nahi nikla. Shayad scanned (image wali) PDF hai."
        )

    return pages