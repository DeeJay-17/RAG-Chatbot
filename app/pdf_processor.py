# FILE: app/pdf_processor.py

import uuid
from sqlalchemy import MetaData
import tiktoken
import pymupdf4llm

from app.models import get_pdf_chunksets_table

def chunk_fixed_tokens(
    text: str,
    chunk_tokens: int = 200,
    overlap_tokens: int = 20,
    encoding_name: str = "cl100k_base"
):
    if not text:
        return []
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be < chunk_tokens")

    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)

    chunks = []
    start = 0
    n = len(tokens)

    while start < n:
        end = min(start + chunk_tokens, n)
        chunk_text = enc.decode(tokens[start:end]).strip()
        if chunk_text:
            chunks.append({"text": chunk_text, "token_start": start, "token_end": end})

        if end == n:
            break

        start = end - overlap_tokens

    return chunks


def is_markdown_header(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False

    # #, ##, ###...
    if s.startswith("#"):
        return True

    # **Header**
    if s.startswith("**") and s.endswith("**") and len(s) > 4:
        return True

    return False


def normalize_header_text(line: str) -> str:
    s = (line or "").strip()

    while s.startswith("#"):
        s = s[1:].strip()

    if s.startswith("**") and s.endswith("**") and len(s) > 4:
        s = s[2:-2].strip()

    return s


def token_count(text: str, encoding_name: str = "cl100k_base") -> int:
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text or ""))


def split_text_by_tokens(text: str, max_tokens: int, overlap_tokens: int, encoding_name: str = "cl100k_base"):
    if not text:
        return []

    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be < max_tokens")

    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)

    parts = []
    start = 0
    n = len(tokens)

    while start < n:
        end = min(start + max_tokens, n)
        part = enc.decode(tokens[start:end]).strip()
        if part:
            parts.append(part)

        if end == n:
            break

        start = end - overlap_tokens

    return parts


def chunk_by_markdown_headers_tokens(
    markdown_text: str,
    min_tokens: int = 120,
    max_tokens: int = 500,
    overlap_tokens: int = 60,
    encoding_name: str = "cl100k_base"
):
    if not markdown_text:
        return []

    lines = markdown_text.splitlines()

    chunks = []
    current_header = None
    current_lines = []

    def flush_section():
        nonlocal current_lines
        txt = "\n".join(current_lines).strip()
        if txt:
            chunks.append({"header": current_header, "text": txt})
        current_lines = []

    for line in lines:
        if is_markdown_header(line):
            flush_section()
            current_header = normalize_header_text(line)
        else:
            if line.strip():
                current_lines.append(line)

    flush_section()

    # --- Merge small chunks based on token count ---
    merged = []
    buf = None

    for c in chunks:
        if buf is None:
            buf = c
            continue

        if token_count(buf.get("text", ""), encoding_name) < min_tokens:
            buf["text"] = (buf["text"] + "\n" + c["text"]).strip()
        else:
            merged.append(buf)
            buf = c

    if buf:
        merged.append(buf)

    final_chunks = []
    for c in merged:
        txt = c.get("text", "")
        hdr = c.get("header")

        if token_count(txt, encoding_name) <= max_tokens:
            final_chunks.append(c)
        else:
            parts = split_text_by_tokens(
                txt,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
                encoding_name=encoding_name
            )
            for part in parts:
                final_chunks.append({"header": hdr, "text": part})

    return final_chunks


# -------------------------------
# Add chunk_id into each chunk dict
# -------------------------------
def add_chunk_ids(chunks, doc_id: str, method: str):
    """
    Mutates/returns chunks so each chunk dict has:
      chunk_id = "{doc_id}:{method}:{chunk_index}"
    """
    out = []
    for i, ch in enumerate(chunks or []):
        # ensure dict
        if isinstance(ch, dict):
            d = dict(ch)
        else:
            d = {"text": str(ch)}

        d["chunk_id"] = f"{doc_id}:{method}:{i}"
        out.append(d)
    return out


def process_pdf(file_path: str, file_name: str, engine):
    """
    - Convert PDF -> Markdown (PyMuPDF4LLM)
    - Chunk via:
        1) fixed_tokens
        2) header_md
    - Store 2 rows (JSONB) in a dynamic Postgres table named doc_id
    - Each chunk now also includes `chunk_id`
    """
    doc_id = str(uuid.uuid4())
    table_name = doc_id

    md_text = pymupdf4llm.to_markdown(file_path) or ""

    fixed_chunks = chunk_fixed_tokens(md_text, chunk_tokens=200, overlap_tokens=20)
    header_chunks = chunk_by_markdown_headers_tokens(
        md_text,
        min_tokens=120,
        max_tokens=500,
        overlap_tokens=60,
        encoding_name="cl100k_base"
    )

    # ✅ add chunk_id for readability / labeling
    fixed_chunks = add_chunk_ids(fixed_chunks, doc_id=doc_id, method="fixed_tokens")
    header_chunks = add_chunk_ids(header_chunks, doc_id=doc_id, method="header_md")

    md = MetaData()
    t = get_pdf_chunksets_table(table_name, md)
    md.create_all(engine)

    rows = [
        {"file_name": file_name, "chunking_method": "fixed_tokens", "chunks_json": fixed_chunks},
        {"file_name": file_name, "chunking_method": "header_md", "chunks_json": header_chunks},
    ]

    with engine.begin() as conn:
        conn.execute(t.insert(), rows)

    return doc_id, table_name
