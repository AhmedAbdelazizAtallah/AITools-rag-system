import io
import re
from typing import List, Dict, Any
from pypdf import PdfReader
from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

def clean_text(text: str) -> str:
    """Clean extracted text by removing extra whitespaces and fixing formatting."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def parse_pdf_bytes(file_bytes: bytes, file_name: str) -> List[Dict[str, Any]]:
    """Reads a PDF from memory and extracts text page by page."""
    pdf_file = io.BytesIO(file_bytes)
    reader = PdfReader(pdf_file)
    extracted_pages = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            extracted_pages.append({
                "page_number": page_num,
                "text": clean_text(text)
            })

    return extracted_pages

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Splits text into overlapping word-based chunks to preserve context."""
    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += (chunk_size - overlap)

    return chunks

def process_uploaded_files(
    uploaded_files: List[Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Dict[str, Any]]:
    """Main pipeline: Read, clean, and chunk uploaded PDF files."""
    documents = []
    doc_id = 0

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        file_name = uploaded_file.name
        uploaded_file.seek(0)

        pages = parse_pdf_bytes(file_bytes, file_name)

        for page_data in pages:
            page_number = page_data["page_number"]
            page_text = page_data["text"]

            chunks = chunk_text(page_text, chunk_size, overlap)

            for chunk_idx, chunk_content in enumerate(chunks):
                documents.append({
                    "id": doc_id,
                    "text": chunk_content,
                    "metadata": {
                        "source_file": file_name,
                        "page_number": page_number,
                        "chunk_index": chunk_idx
                    }
                })
                doc_id += 1

    return documents