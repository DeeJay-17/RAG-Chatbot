# FILE: main.py

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import re
import json
from sqlalchemy import insert, select, desc, text
from app.db import engine
from app.models import metadata, master_files
from app.pdf_processor import process_pdf
from app.vector_sync import sync_pdf_table_to_chroma
from app.chroma_client import get_chroma_http_client, get_or_create_collection
from typing import List, Dict, Any
from pydantic import BaseModel
from app.eval_metrics import recall_at_k, mrr_at_k
from app.llm_client import chat_completion, LLMError
from app.sql_safety import ensure_safe_select
from app.csv_processor import process_csv
from app.prompts import (
    normalize_prompt_style,
    pdf_answer_system_prompt,
    csv_postprocess_system_prompt,
    chat_route_system_prompt,
    chat_csv_sql_system_prompt,
    chat_answer_csv_sql_system_prompt,
)
from app.pii import mask_text, mask_rows
from app.tracing import get_request_id, start_timer, elapsed_ms, persist_trace

app = FastAPI()
metadata.create_all(engine)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
MASTER_CATALOG_PATH = os.path.join(UPLOAD_DIR, "master_catalog.json")


def _load_master_catalog() -> dict:
    if not os.path.exists(MASTER_CATALOG_PATH):
        return {"tables": []}
    try:
        with open(MASTER_CATALOG_PATH, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            return {"tables": []}
        tables = obj.get("tables")
        if not isinstance(tables, list):
            obj["tables"] = []
        return obj
    except Exception:
        # If corrupt, don't crash the app—start fresh.
        return {"tables": []}


def _append_to_master_catalog(*, table_name: str, file_name: str, catalog: dict) -> None:
    master = _load_master_catalog()
    tables = master.get("tables") or []
    # Remove any existing entry for the same table_name (idempotent-ish)
    tables = [t for t in tables if isinstance(t, dict) and t.get("table_name") != table_name]

    entry = {
        "table_name": table_name,
        "file_name": file_name,
        "overall_description": catalog.get("overall_description", "") if isinstance(catalog, dict) else "",
        "columns": (catalog.get("columns") if isinstance(catalog, dict) else None),
    }

    if entry["columns"] is None:
        entry["columns"] = catalog

    tables.append(entry)
    master["tables"] = tables

    tmp = MASTER_CATALOG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(master, f, ensure_ascii=False, indent=2)
    os.replace(tmp, MASTER_CATALOG_PATH)

def parse_json_strict(raw: str) -> dict:
    if not raw:
        raise ValueError("Empty model output")

    s = raw.strip()

    s = re.sub(r"^```json\s*", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"^```\s*", "", s).strip()
    s = re.sub(r"\s*```$", "", s).strip()

    return json.loads(s)


@app.get("/chroma/health")
def chroma_health():
    client = get_chroma_http_client()
    hb = client.heartbeat()
    return {"ok": True, "heartbeat": hb}


@app.get("/chroma/count")
def chroma_count():
    client = get_chroma_http_client()
    col = get_or_create_collection(client)
    return {"collection": col.name, "count": col.count()}


@app.post("/upload/")
def upload_file(
    file: UploadFile = File(...),
    catalog: UploadFile | None = File(None),
    description: str | None = Form(None) 
):
    file_path = f"{UPLOAD_DIR}/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    fname = file.filename.lower()

    # -----------------------------
    # PDF upload
    # -----------------------------
    if fname.endswith(".pdf"):
        doc_id, table_name = process_pdf(file_path, file.filename, engine)
        
        # Use provided description or fallback
        final_desc = description if description and description.strip() else f"Uploaded PDF file: {file.filename}"

        with engine.begin() as conn:
            conn.execute(insert(master_files), {
                "doc_id": doc_id,
                "file_name": file.filename,
                "file_type": "pdf",
                "table_name": table_name,
                "description": final_desc
            })

        sync_result = sync_pdf_table_to_chroma(table_name=table_name, doc_id=doc_id)

        return {
            "message": "PDF uploaded",
            "doc_id": doc_id,
            "file_type": "pdf",
            "table": table_name,
            "chroma_sync": sync_result,
            "description_saved": final_desc
        }

    # -----------------------------
    # CSV upload
    # -----------------------------
    if fname.endswith(".csv"):
        if catalog is None:
            raise HTTPException(
                status_code=400,
                detail="CSV upload requires a catalog JSON file uploaded as form field `catalog`.",
            )
        try:
            raw = catalog.file.read()
            cat_obj = json.loads(raw.decode("utf-8"))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid catalog JSON: {e}")

        doc_id, table_name = process_csv(file_path, file.filename, engine)
        
        # Extract description from Catalog
        catalog_desc = cat_obj.get("overall_description", "")
        final_desc = catalog_desc if catalog_desc else (description or f"Uploaded CSV file: {file.filename}")

        with engine.begin() as conn:
            conn.execute(insert(master_files), {
                "doc_id": doc_id,
                "file_name": file.filename,
                "file_type": "csv",
                "table_name": table_name,
                "description": final_desc
            })

        _append_to_master_catalog(table_name=table_name, file_name=file.filename, catalog=cat_obj)

        return {
            "message": "CSV uploaded",
            "doc_id": doc_id,
            "file_type": "csv",
            "table": table_name,
            "description_saved": final_desc
        }

    raise HTTPException(status_code=400, detail="Unsupported file format (only PDF and CSV)")


@app.get("/datasets/{table_name}/csv_preview")
def preview_csv_table(table_name: str, limit: int = 50):
    # your table_name is a UUID string (doc_id)
    if not re.fullmatch(r"[0-9a-fA-F-]{36}", table_name):
        raise HTTPException(status_code=400, detail="Invalid table name")

    limit = max(1, min(int(limit), 10000))

    with engine.connect() as conn:
        res = conn.execute(text(f'SELECT * FROM "{table_name}" LIMIT {limit}'))
        cols = list(res.keys())
        rows = [dict(zip(cols, r)) for r in res.fetchall()]

    return {"columns": cols, "rows": rows}


@app.get("/datasets/{table_name}/preview")
def preview_pdf_chunksets(table_name: str):
    if not re.fullmatch(r"[0-9a-fA-F-]{36}", table_name):
        raise HTTPException(status_code=400, detail="Invalid table name")

    with engine.connect() as conn:
        res = conn.execute(text(f'SELECT id, file_name, chunking_method, chunks_json, created_at FROM "{table_name}"'))
        cols = list(res.keys())
        data = [dict(zip(cols, row)) for row in res.fetchall()]

    return {"columns": cols, "rows": data}


@app.get("/datasets/csv")
def list_csv_datasets():
    with engine.connect() as conn:
        res = conn.execute(
            select(
                master_files.c.doc_id,
                master_files.c.file_name,
                master_files.c.file_type,
                master_files.c.table_name,
                master_files.c.created_at
            ).where(master_files.c.file_type == "csv").order_by(desc(master_files.c.created_at))
        )
        rows = [dict(r._mapping) for r in res.fetchall()]
    return rows


@app.get("/search/pdf")
def search_pdf(query: str, k: int = 5):
    client = get_chroma_http_client()
    col = get_or_create_collection(client)

    methods = ["fixed_tokens", "header_md"]
    out = {"query": query, "k": k, "results": {}}

    for m in methods:
        res = col.query(
            query_texts=[query],
            n_results=k,
            where={"chunking_method": m},
            include=["documents", "distances", "metadatas"],
        )

        docs = res.get("documents", [[]])[0]
        dists = res.get("distances", [[]])[0]
        metas = res.get("metadatas", [[]])[0]

        items = []
        for i in range(len(docs)):
            items.append({
                "text": docs[i],
                "distance": dists[i] if i < len(dists) else None,
                "metadata": metas[i] if i < len(metas) else None,
            })

        out["results"][m] = items

    return out


class EvalQuery(BaseModel):
    query: str
    doc_id: str
    relevant_chunk_ids: List[str]

class EvalRequest(BaseModel):
    queries: List[EvalQuery]
    k: int = 5

@app.post("/eval/pdf")
def eval_pdf(req: EvalRequest):
    client = get_chroma_http_client()
    col = get_or_create_collection(client)

    methods = ["fixed_tokens", "header_md"]
    k = int(req.k)

    per_method = {m: {"recall": [], "mrr": []} for m in methods}
    details = []

    for q in req.queries:
        relevant_set = set(q.relevant_chunk_ids or [])

        row = {"query": q.query, "doc_id": q.doc_id, "per_method": {}}

        for m in methods:
            res = col.query(
                query_texts=[q.query],
                n_results=k,
                where={"$and": [{"chunking_method": m}, {"doc_id": q.doc_id}]},
                include=["documents", "distances", "metadatas"],
            )


            docs = res.get("documents", [[]])[0]
            dists = res.get("distances", [[]])[0]
            metas = res.get("metadatas", [[]])[0]

            retrieved = []
            for i in range(len(docs)):
                md = metas[i] if i < len(metas) else {}
                chunk_index = md.get("chunk_index")
                doc_id = md.get("doc_id")
                chunk_method = md.get("chunking_method")
                chunk_id = None
                if doc_id is not None and chunk_method is not None and chunk_index is not None:
                    chunk_id = f"{doc_id}:{chunk_method}:{chunk_index}"

                retrieved.append({
                    "id": chunk_id,
                    "text": docs[i],
                    "distance": dists[i] if i < len(dists) else None,
                    "metadata": md,
                })

            def is_relevant(item: Dict[str, Any]) -> bool:
                return (item.get("id") in relevant_set)

            r = recall_at_k(retrieved, is_relevant, k)
            mrr = mrr_at_k(retrieved, is_relevant, k)

            per_method[m]["recall"].append(r)
            per_method[m]["mrr"].append(mrr)

            row["per_method"][m] = {"recall": r, "mrr": mrr}

        details.append(row)

    summary = {}
    for m in methods:
        recalls = per_method[m]["recall"]
        mrrs = per_method[m]["mrr"]
        summary[m] = {
            "recall_at_k": sum(recalls) / len(recalls) if recalls else 0.0,
            "mrr_at_k": sum(mrrs) / len(mrrs) if mrrs else 0.0,
            "n": len(recalls),
            "k": k
        }

    return {"k": k, "summary": summary, "details": details}


class RouteRequest(BaseModel):
    query: str

class RouteResponse(BaseModel):
    route: str  # "pdf" or "csv"
    reasoning: str | None = None

class CsvSqlRequest(BaseModel):
    query: str
    catalog: Dict[str, Any]
    table_name: str

class CsvSqlResponse(BaseModel):
    sql: str
    columns: List[str]
    rows: List[Dict[str, Any]]


class ChatAnswerResponse(BaseModel):
    route: str  # "pdf" or "csv"
    answer: str
    debug: Dict[str, Any] | None = None


@app.post("/chat/route", response_model=RouteResponse)
def chat_route(req: RouteRequest):

    try:
        with engine.connect() as conn:
            # We select file_type, file_name, and description
            stmt = select(master_files.c.file_type, master_files.c.file_name, master_files.c.description)
            res = conn.execute(stmt)
            rows = res.fetchall()
        
        context_lines = []
        if not rows:
            context_lines.append("No files available.")
        else:
            for r in rows:
                ftype = r.file_type.upper() if r.file_type else "UNKNOWN"
                fname = r.file_name
                desc_text = r.description if r.description else "No description provided."
                context_lines.append(f"- [{ftype}] File: '{fname}': {desc_text}")
        
        file_context_str = "\n".join(context_lines)

    except Exception as e:

        file_context_str = f"Error fetching file context: {str(e)}"

    system = {
        "role": "system",
        "content": chat_route_system_prompt(file_context_str),
    }
    user = {"role": "user", "content": req.query}

    try:
        raw = chat_completion([system, user], temperature=0.0)
    except LLMError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # import json
    try:
        obj = parse_json_strict(raw)
        route = obj.get("route")
        if route not in ("pdf", "csv"):
            raise ValueError("Invalid route")
        return {"route": route, "reasoning": obj.get("reasoning")}
    except Exception:
        # fallback heuristic
        q = (req.query or "").lower()
        if any(w in q for w in ["table", "csv", "column", "average", "sum", "group by", "filter", "rows"]):
            return {"route": "csv", "reasoning": "fallback_heuristic"}
        return {"route": "pdf", "reasoning": "fallback_heuristic"}


@app.post("/chat/csv_sql", response_model=CsvSqlResponse)
def chat_csv_sql(req: CsvSqlRequest):
    """
    Use catalog+question to generate SQL, validate, execute, return results.
    """
    payload = {
        "user_query": req.query,
        "csv_overall_description": req.catalog.get("overall_description", ""),
        "columns": req.catalog.get("columns", req.catalog),
        "table_name": req.table_name
    }

    system = {
        "role": "system",
        "content": chat_csv_sql_system_prompt(),
    }
    user = {
        "role": "user",
        "content": (
            "Given this data catalog + user query, write the SQL.\n"
            f"{payload}"
        )
    }

    try:
        raw = chat_completion([system, user], temperature=0.0)
    except LLMError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # import json
    try:
        obj = parse_json_strict(raw)
        sql = obj["sql"]
    except Exception:
        raise HTTPException(status_code=500, detail=f"LLM did not return valid JSON: {raw}")

    # Safety
    try:
        safe_sql = ensure_safe_select(sql)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unsafe SQL rejected: {e}")

    # Execute
    with engine.connect() as conn:
        res = conn.execute(text(safe_sql))
        cols = list(res.keys())
        rows = [dict(zip(cols, r)) for r in res.fetchall()]

    return {"sql": safe_sql, "columns": cols, "rows": rows}


class ChatAnswerRequest(BaseModel):
    query: str
    prompt_style: str | None = "zero_shot"


@app.post("/chat/answer", response_model=ChatAnswerResponse)
def chat_answer(req: ChatAnswerRequest, request: Request):

    style = normalize_prompt_style(req.prompt_style)

    # Initialize trace context (LangChain-style state + events)
    request_id = get_request_id(request)
    t0 = start_timer()
    events: list[dict[str, Any]] = []
    state: dict[str, Any] = {"prompt_style": style}

    # 1) route
    route_obj = chat_route(RouteRequest(query=req.query))
    route = route_obj.get("route", "pdf")
    events.append({"type": "route_decision", "route": route, "raw": route_obj})
    state["route"] = route

    # Common trace fields populated as the pipeline progresses
    retrieved_docs = None
    llm_prompt_text = None
    llm_prompt_messages: list[dict[str, Any]] | None = None
    llm_answer = None
    embeddings_meta = None

    if route == "pdf":
        try:
            res = search_pdf(query=req.query, k=1)
        except Exception as e:
            # On hard failure we still want to know we entered PDF path.
            duration = elapsed_ms(t0)
            persist_trace(
                request_id=request_id,
                endpoint="/chat/answer",
                http_method="POST",
                status_code=500,
                user_query=req.query,
                route="pdf",
                retrieved_docs_with_scores=None,
                llm_prompt=None,
                llm_prompt_messages=None,
                llm_response=None,
                embeddings_metadata=None,
                state=state,
                events=events + [{"type": "error", "stage": "pdf_search", "error": str(e)}],
                request_payload={"query": req.query, "prompt_style": req.prompt_style},
                response_body=None,
                error_message=f"PDF search failed: {e}",
                duration_ms=duration,
            )
            raise HTTPException(status_code=500, detail=f"PDF search failed: {e}")

        results = (res.get("results") or {})
        retrieved_docs = res
        fixed = (results.get("fixed_tokens") or [])[:1]
        header = (results.get("header_md") or [])[:1]

        # Decide the single best chunk across both methods (smallest distance).
        best_chunk = None
        best_source = None  # "fixed_tokens" or "header_md"

        def _update_best(source_name: str, items: list[dict]):
            nonlocal best_chunk, best_source
            if not items:
                return
            it = items[0]
            dist = it.get("distance")
            if dist is None:
                return
            if best_chunk is None or (best_chunk.get("distance") is not None and dist < best_chunk.get("distance")):
                best_chunk = it
                best_source = source_name

        _update_best("fixed_tokens", fixed)
        _update_best("header_md", header)

        masked_chunk = dict(best_chunk or {})
        if "text" in masked_chunk:
            masked_chunk["text"] = mask_text(masked_chunk.get("text"))

        context = {
            "query": req.query,
            "chosen_source": best_source,
            "chosen_chunk": masked_chunk,
        }
        state["pdf_context"] = context
        embeddings_meta = results

        system = {
            "role": "system",
            "content": pdf_answer_system_prompt(style),
        }
        user = {"role": "user", "content": f"{context}"}
        llm_prompt_messages = [system, user]
        llm_prompt_text = f"SYSTEM:\n{system['content']}\n\nUSER:\n{user['content']}"
        try:
            raw = chat_completion([system, user], temperature=0.2)
            obj = parse_json_strict(raw)
            ans = obj.get("answer") or ""
        except Exception:
            ans = "I couldn’t structure an answer from the retrieved PDF chunks."

        llm_answer = ans
        events.append({"type": "llm_call", "mode": "pdf_answer"})
        duration = elapsed_ms(t0)
        persist_trace(
            request_id=request_id,
            endpoint="/chat/answer",
            http_method="POST",
            status_code=200,
            user_query=req.query,
            route="pdf",
            retrieved_docs_with_scores=retrieved_docs,
            llm_prompt=llm_prompt_text,
            llm_prompt_messages=llm_prompt_messages,
            llm_response=llm_answer,
            embeddings_metadata=embeddings_meta,
            state=state,
            events=events,
            request_payload={"query": req.query, "prompt_style": req.prompt_style},
            response_body={"route": "pdf", "answer": llm_answer},
            duration_ms=duration,
        )

        return {
            "route": "pdf",
            "answer": llm_answer,
            "debug": {
                "retrieval": context,
                "pdf_top1": {
                    "fixed_tokens": fixed,
                    "header_md": header,
                },
            },
        }

    # CSV route
    master = _load_master_catalog()
    tables = master.get("tables") or []
    if not tables:
        duration = elapsed_ms(t0)
        persist_trace(
            request_id=request_id,
            endpoint="/chat/answer",
            http_method="POST",
            status_code=200,
            user_query=req.query,
            route="csv",
            retrieved_docs_with_scores=None,
            llm_prompt=None,
            llm_prompt_messages=None,
            llm_response="No CSV catalogs are available yet. Upload a CSV along with its catalog JSON first.",
            embeddings_metadata=None,
            state=state,
            events=events + [{"type": "csv_catalog", "status": "empty"}],
            request_payload={"query": req.query, "prompt_style": req.prompt_style},
            response_body={
                "route": "csv",
                "answer": "No CSV catalogs are available yet. Upload a CSV along with its catalog JSON first.",
                "debug": {"reason": "empty_master_catalog"},
            },
            duration_ms=duration,
        )
        return {
            "route": "csv",
            "answer": "No CSV catalogs are available yet. Upload a CSV along with its catalog JSON first.",
            "debug": {"reason": "empty_master_catalog"},
        }

    state["master_catalog_size"] = len(tables)

    # 2) Generate SQL that can use multiple tables based on the full master catalog
    sql_system = {
        "role": "system",
        "content": chat_answer_csv_sql_system_prompt(),
    }
    sql_user = {
        "role": "user",
        "content": (
            "User query:\n"
            f"{req.query}\n\n"
            "Master catalog of available tables:\n"
            f"{master}"
        ),
    }
    events.append({"type": "llm_call", "stage": "csv_sql_generation"})
    try:
        raw_sql = chat_completion([sql_system, sql_user], temperature=0.0)
        obj_sql = parse_json_strict(raw_sql)
        sql = obj_sql["sql"]
    except Exception:
        duration = elapsed_ms(t0)
        persist_trace(
            request_id=request_id,
            endpoint="/chat/answer",
            http_method="POST",
            status_code=200,
            user_query=req.query,
            route="csv",
            retrieved_docs_with_scores=None,
            llm_prompt=None,
            llm_prompt_messages=[sql_system, sql_user],
            llm_response=None,
            embeddings_metadata=None,
            state=state,
            events=events + [{"type": "error", "stage": "csv_sql_generation"}],
            request_payload={"query": req.query, "prompt_style": req.prompt_style},
            response_body={
                "route": "csv",
                "answer": f"LLM did not return valid SQL JSON for your question. Raw output was: {raw_sql!r}",
                "debug": {"reason": "sql_generation_failed"},
            },
            error_message="LLM did not return valid SQL JSON for your question.",
            duration_ms=duration,
        )
        return {
            "route": "csv",
            "answer": f"LLM did not return valid SQL JSON for your question. Raw output was: {raw_sql!r}",
            "debug": {"reason": "sql_generation_failed"},
        }

    # Safety + execution
    try:
        safe_sql = ensure_safe_select(sql)
    except Exception as e:
        duration = elapsed_ms(t0)
        persist_trace(
            request_id=request_id,
            endpoint="/chat/answer",
            http_method="POST",
            status_code=200,
            user_query=req.query,
            route="csv",
            retrieved_docs_with_scores=None,
            llm_prompt=None,
            llm_prompt_messages=None,
            llm_response=None,
            embeddings_metadata=None,
            state=state,
            events=events + [{"type": "sql_validation", "success": False, "error": str(e)}],
            request_payload={"query": req.query, "prompt_style": req.prompt_style},
            response_body={
                "route": "csv",
                "answer": f"Proposed SQL was rejected as unsafe: {e}",
                "debug": {"reason": "unsafe_sql", "sql": sql},
            },
            error_message=f"Proposed SQL was rejected as unsafe: {e}",
            duration_ms=duration,
        )
        return {
            "route": "csv",
            "answer": f"Proposed SQL was rejected as unsafe: {e}",
            "debug": {"reason": "unsafe_sql", "sql": sql},
        }

    with engine.connect() as conn:
        res = conn.execute(text(safe_sql))
        cols = list(res.keys())
        rows = [dict(zip(cols, r)) for r in res.fetchall()]

    # 3) Post-process to natural language answer
    # Mask PII in result rows before sending to the LLM
    masked_rows = mask_rows(rows)
    post_system = {
        "role": "system",
        "content": csv_postprocess_system_prompt(style),
    }
    post_user = {
        "role": "user",
        "content": (
            f"User question:\n{req.query}\n\n"
            f"SQL:\n{safe_sql}\n\n"
            f"Columns:\n{cols}\n\n"
            f"Rows (up to 200, PII-masked):\n{masked_rows}"
        ),
    }
    llm_prompt_messages = [post_system, post_user]
    llm_prompt_text = f"SYSTEM:\n{post_system['content']}\n\nUSER:\n{post_user['content']}"
    events.append({"type": "llm_call", "stage": "csv_postprocess"})
    try:
        raw_post = chat_completion([post_system, post_user], temperature=0.2)
        post = parse_json_strict(raw_post)
        answer = post.get("answer") or ""
    except Exception:
        # Basic fallback if model output isn't JSON
        if not rows:
            answer = "No matching rows were found for your question."
        else:
            answer = "I found results for your question, but couldn't format them nicely."

    llm_answer = answer
    state["sql"] = safe_sql
    state["n_rows"] = len(rows)
    duration = elapsed_ms(t0)
    persist_trace(
        request_id=request_id,
        endpoint="/chat/answer",
        http_method="POST",
        status_code=200,
        user_query=req.query,
        route="csv",
        retrieved_docs_with_scores=None,
        llm_prompt=llm_prompt_text,
        llm_prompt_messages=llm_prompt_messages,
        llm_response=llm_answer,
        embeddings_metadata=None,
        state=state,
        events=events,
        request_payload={"query": req.query, "prompt_style": req.prompt_style},
        response_body={
            "route": "csv",
            "answer": llm_answer,
            "debug": {
                "sql": safe_sql,
                "n_rows": len(rows),
            },
        },
        duration_ms=duration,
    )

    return {
        "route": "csv",
        "answer": llm_answer,
        "debug": {
            "sql": safe_sql,
            "n_rows": len(rows),
        },
    }