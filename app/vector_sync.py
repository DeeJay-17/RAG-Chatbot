# ============================================
# FILE: app/vector_sync.py
# ============================================
import re
from sqlalchemy import text
from app.db import engine
from app.chroma_client import get_chroma_http_client, get_or_create_collection


def sync_pdf_table_to_chroma(table_name: str, doc_id: str):
    """
    Beginner-friendly sync:
    - Read 2 rows from Postgres (fixed_tokens, header_md)
    - Upsert ALL chunks into Chroma (one upsert per method)
    """
    if not re.fullmatch(r"[0-9a-fA-F-]{36}", table_name):
        raise ValueError("Invalid table name")

    sql = text(f'SELECT file_name, chunking_method, chunks_json FROM "{table_name}"')

    with engine.connect() as conn:
        rows = conn.execute(sql).mappings().all()

    client = get_chroma_http_client()
    col = get_or_create_collection(client)

    upserted = 0

    for r in rows:
        method = r["chunking_method"]
        file_name = r["file_name"]
        chunks = r["chunks_json"] or []

        ids, docs, metas = [], [], []

        for i, ch in enumerate(chunks):
            txt = (ch.get("text") if isinstance(ch, dict) else str(ch)) or ""
            if not txt.strip():
                continue

            ids.append(f"{doc_id}:{method}:{i}")
            docs.append(txt)
            metas.append({
                "doc_id": doc_id,
                "table_name": table_name,
                "file_name": file_name,
                "chunking_method": method,
                "chunk_index": i,
            })

        if ids:
            col.upsert(ids=ids, documents=docs, metadatas=metas)
            upserted += len(ids)

    return {"doc_id": doc_id, "table_name": table_name, "upserted": upserted}
