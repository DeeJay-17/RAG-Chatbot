# FILE: app/models.py

from sqlalchemy import Table, Column, Integer, String, MetaData, DateTime, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime

metadata = MetaData()


master_files = Table(
    "master_files",
    metadata,
    Column("doc_id", String, primary_key=True),
    Column("file_name", String, nullable=False),
    Column("file_type", String, nullable=False),
    Column("table_name", String, nullable=False),
    Column("description", Text),
    Column("created_at", DateTime, server_default=text("NOW()")),
)


def get_pdf_chunksets_table(table_name: str, md: MetaData) -> Table:
    return Table(
        table_name,
        md,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("file_name", String, nullable=False),
        Column("chunking_method", String, nullable=False),
        Column("chunks_json", JSONB, nullable=False),
        Column("created_at", DateTime, default=datetime.utcnow),
    )


llm_traces = Table(
    "llm_traces",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("request_id", String, nullable=False, index=True),
    Column("endpoint", String, nullable=False),
    Column("http_method", String, nullable=False),
    Column("status_code", Integer, nullable=True),
    Column("user_query", Text, nullable=True),
    Column("route", String, nullable=True),
    Column("retrieved_docs_with_scores", JSONB, nullable=True),
    Column("llm_prompt", Text, nullable=True),
    Column("llm_prompt_messages", JSONB, nullable=True),
    Column("llm_response", Text, nullable=True),
    Column("embeddings_metadata", JSONB, nullable=True),
    Column("state", JSONB, nullable=True),
    Column("events", JSONB, nullable=True),
    Column("request_payload", JSONB, nullable=True),
    Column("response_body", JSONB, nullable=True),
    Column("error_message", Text, nullable=True),
    Column("duration_ms", Integer, nullable=True),
    Column("created_at", DateTime, server_default=text("NOW()")),
)
