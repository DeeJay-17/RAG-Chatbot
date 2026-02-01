# ============================================
# FILE: app/models.py
# ============================================
from sqlalchemy import Table, Column, Integer, String, MetaData, DateTime, text
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime

metadata = MetaData()

from sqlalchemy import Column, String, Text, DateTime

master_files = Table(
    "master_files",
    metadata,
    Column("doc_id", String, primary_key=True),
    Column("file_name", String, nullable=False),
    Column("file_type", String, nullable=False),  # pdf | csv
    Column("table_name", String, nullable=False),
    Column("description", Text),  # ✅ NEW
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
