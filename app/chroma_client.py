# FILE: app/chroma_client.py

import os
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

load_dotenv()

def get_chroma_http_client() -> chromadb.HttpClient:
    host = os.getenv("CHROMA_HOST", "localhost")
    port = int(os.getenv("CHROMA_PORT", "8000"))
    return chromadb.HttpClient(
        host=host,
        port=port,
        settings=Settings(allow_reset=False),
    )

def get_or_create_collection(client: chromadb.HttpClient):
    name = os.getenv("CHROMA_COLLECTION", "pdf_chunks")
    embed_fn = ONNXMiniLM_L6_V2()
    return client.get_or_create_collection(
        name=name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
