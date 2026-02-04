# streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import requests
from sqlalchemy import select, desc  # only needed if you keep DB reads

from app.db import engine
from app.models import master_files
from app.styles import Themes

from dotenv import load_dotenv
load_dotenv()


st.set_page_config(page_title="Data Ingestion Portal", layout="wide")

FASTAPI_URL = os.getenv("FASTAPI_URL")

def apply_theme(theme_name):
    css = Themes.get_css(theme_name)
    if css:
        st.markdown(
            """
            <style>
            /* Streamlit JSON / code blocks inside expanders */
            div[data-testid="stJson"] pre,
            div[data-testid="stJson"] span,
            div[data-testid="stCodeBlock"] pre,
            div[data-testid="stCodeBlock"] span {
                color: #FFFFFF !important;
                background-color: #0E1117 !important;
            }

            /* Expander content background */
            details > div {
                background-color: #0E1117 !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )


# --- PAGE 1: UPLOAD (BLUE THEME) ---
def show_upload_page():
    apply_theme("upload")

    st.title("📂 File Ingestion Portal")
    st.markdown("### Upload Files")

    uploaded_file = st.file_uploader("Choose a file (PDF or CSV)", type=['csv', 'pdf'])
    
    # Logic for Catalog or Description inputs
    uploaded_catalog = None
    user_description = ""

    if uploaded_file is not None:
        file_ext = uploaded_file.name.lower().split('.')[-1]
        
        if file_ext == "csv":
            st.markdown("### CSV Catalog (required)")
            st.caption(
                "Upload a catalog JSON for this CSV. It will be appended to the master catalog used by the chatbot.\n\n"
                "Recommended format:\n"
                "{\n"
                '  "overall_description": "...",\n'
                '  "columns": [\n'
                '    {"column":"...", "data_type":"...", "description":"..."},\n'
                "    ...\n"
                "  ]\n"
                "}"
            )
            uploaded_catalog = st.file_uploader("Catalog JSON", type=["json"], key="upload_catalog_json")
        
        elif file_ext == "pdf":
            st.markdown("### PDF Description (required)")
            st.caption("Provide a brief description of the PDF content to help the chatbot route questions correctly.")
            user_description = st.text_input("Description:", placeholder="e.g. Employee Handbook 2024, Q3 Financial Report")

    if uploaded_file is not None:
        if st.button("Process & Ingest File"):
            if uploaded_file.name.lower().endswith(".pdf") and not user_description.strip():
                st.error("Please provide a description for the PDF.")
                return

            with st.spinner(f"Uploading {uploaded_file.name} via FastAPI..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    data_payload = {}

                    if uploaded_file.name.lower().endswith(".csv"):
                        if uploaded_catalog is None:
                            st.error("Please upload the catalog JSON for this CSV.")
                            return
                        files["catalog"] = (uploaded_catalog.name, uploaded_catalog.getvalue())
                    
                    if user_description:
                        data_payload["description"] = user_description

                    # Send files AND description
                    resp = requests.post(f"{FASTAPI_URL}/upload/", files=files, data=data_payload, timeout=300)

                    if resp.status_code != 200:
                        st.error(f"FastAPI error ({resp.status_code}): {resp.text}")
                        return

                    data = resp.json()
                    st.success(f"{data.get('message')} | Doc ID: {data.get('doc_id')}")

                    if data.get("file_type") == "csv":
                        st.session_state["last_uploaded_csv_table"] = data.get("table")
                        st.session_state["last_uploaded_csv_name"] = uploaded_file.name
                        st.info(f"Stored in table: {data.get('table')}")
                    
                    if data.get("description_saved"):
                        st.caption(f"Saved description: {data.get('description_saved')}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Could not reach FastAPI at {FASTAPI_URL}. Error: {e}")

# --- PAGE 2: ANALYTICS (GREEN THEME) ---
def show_analytics_page():
    apply_theme("analytics")

    st.title("📊 Data Analytics")

    # Fetch datasets from FastAPI
    try:
        resp = requests.get(f"{FASTAPI_URL}/datasets/csv", timeout=30)
        if resp.status_code != 200:
            st.error(f"Failed to fetch datasets: {resp.text}")
            return
        datasets = resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach FastAPI at {FASTAPI_URL}. Error: {e}")
        return

    if not datasets:
        st.warning("No CSV data found.")
        return

    # map display -> table_name
    dataset_map = {}
    for d in datasets:
        table_uuid = d["table_name"]
        file_name = d["file_name"]
        display_label = f"{file_name} ({table_uuid[:8]})"
        dataset_map[display_label] = table_uuid

    options = list(dataset_map.keys())

    default_idx = 0
    if "last_uploaded_csv_name" in st.session_state:
        for i, opt in enumerate(options):
            if st.session_state["last_uploaded_csv_name"] in opt:
                default_idx = i
                break

    selected_option = st.selectbox("Select Dataset:", options, index=default_idx)
    real_table_name = dataset_map[selected_option]

    # Preview via FastAPI (safe, fast)
    try:
        prev = requests.get(
            f"{FASTAPI_URL}/datasets/{real_table_name}/csv_preview",
            params={"limit": 10000},
            timeout=60
        )

        if prev.status_code != 200:
            st.error(f"Preview fetch failed: {prev.text}")
            return
        payload = prev.json()
        df = pd.DataFrame(payload["rows"])
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch preview from FastAPI: {e}")
        return

    if df.empty:
        st.warning("Dataset is empty.")
        return

    all_columns = df.columns.tolist()

    # --- SECTION 1: DATA PREVIEW ---
    st.subheader("Data Preview (Top 10 Rows)")
    cols_to_show = st.multiselect("Select columns to view:", all_columns, default=all_columns[:5])
    if cols_to_show:
        st.table(df[cols_to_show].head(10))
    else:
        st.info("Select at least one column to view data.")

    st.divider()

    # --- SECTION 2: CHARTS ---
    st.subheader("Visualizations")

    col1, col2, col3 = st.columns(3)
    with col1:
        chart_type = st.selectbox("Chart Type:", ["Bar Chart", "Line Chart", "Scatter Plot", "Box Plot"])
    with col2:
        x_axis = st.selectbox("X Axis:", all_columns, index=0)
    with col3:
        y_axis = st.selectbox("Y Axis:", all_columns, index=1 if len(all_columns) > 1 else 0)

    if st.button("Generate Chart"):
        try:
            green_colors = ['#33691E', '#558B2F', '#7CB342', '#9CCC65']

            if chart_type == "Bar Chart":
                fig = px.bar(df, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}", color_discrete_sequence=green_colors)
            elif chart_type == "Line Chart":
                fig = px.line(df, x=x_axis, y=y_axis, title=f"{y_axis} trend over {x_axis}", color_discrete_sequence=green_colors)
            elif chart_type == "Scatter Plot":
                fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}", color_discrete_sequence=green_colors)
            else:
                fig = px.box(df, x=x_axis, y=y_axis, title=f"Distribution of {y_axis} by {x_axis}", color_discrete_sequence=green_colors)

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error generating chart: {e}. Ensure Y-axis is numeric.")

# --- PAGE 3: DIRECTORY (NEUTRAL THEME) ---
def show_directory_page():
    apply_theme("directory")

    st.title("🗂️ Master Directory")

    # Option A: keep direct DB read (simple)
    with engine.connect() as conn:
        # Assuming 'description' column is added
        query = select(
            master_files.c.doc_id,
            master_files.c.file_name,
            master_files.c.file_type,
            master_files.c.description, # <--- Showing description in directory
            master_files.c.table_name,
            master_files.c.created_at
        ).order_by(desc(master_files.c.created_at))
        df_master = pd.read_sql(query, conn)

    if not df_master.empty:
        st.table(df_master.head(50))
    else:
        st.info("No files found.")


def show_chatbot_page():
    apply_theme("analytics")
    st.title("💬 ChatBot")

    # -----------------------------
    # Session state
    # -----------------------------
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []  # list of dicts: {"role":..., "content":...}
    if "prompt_style" not in st.session_state:
        st.session_state.prompt_style = "zero_shot"

    with st.sidebar:
        if st.button("Clear chat"):
            st.session_state.chat_messages = []
            st.rerun()

    # -----------------------------
    # Render chat history
    # -----------------------------
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # -----------------------------
    # Prompting style selector + Chat input
    # -----------------------------
    style_label = st.selectbox(
        "Prompting style",
        [
            "Zero-shot",
            "One-shot",
            "Few-shot",
            "Chain-of-thought",
            "Stepwise reasoning",
        ],
        index=0,
    )
    style_map = {
        "Zero-shot": "zero_shot",
        "One-shot": "one_shot",
        "Few-shot": "few_shot",
        "Chain-of-thought": "chain_of_thought",
        "Stepwise reasoning": "stepwise",
    }
    st.session_state.prompt_style = style_map.get(style_label, "zero_shot")

    user_q = st.chat_input("Ask a question about your PDFs or CSV datasets...")

    if not user_q:
        return

    # Add user message
    st.session_state.chat_messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    # Decide route via FastAPI
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                ans_resp = requests.post(
                    f"{FASTAPI_URL}/chat/answer",
                    json={
                        "query": user_q,
                        "prompt_style": st.session_state.get("prompt_style", "zero_shot"),
                    },
                    timeout=180,
                )
                if ans_resp.status_code != 200:
                    st.error(ans_resp.text)
                    st.session_state.chat_messages.append({"role": "assistant", "content": f"Error: {ans_resp.text}"})
                    return
                out = ans_resp.json()
            except Exception as e:
                st.error(str(e))
                st.session_state.chat_messages.append({"role": "assistant", "content": f"Error: {e}"})
                return

        answer_text = out.get("answer") or ""
        st.markdown(answer_text)

        # Optional debug + PDF side-by-side comparison (top-1 chunks)
        dbg = out.get("debug") or {}
        pdf_top1 = dbg.get("pdf_top1")
        if pdf_top1:
            fixed = pdf_top1.get("fixed_tokens") or []
            header = pdf_top1.get("header_md") or []

            # st.markdown("#### Top-1 PDF chunks for each chunking method")
            c1, c2 = st.columns(2)

            def top1_box(col, title, items):
                with col:
                    st.subheader(title)
                    if not items:
                        st.info("No result found.")
                        return
                    it = items[0]
                    txt = it.get("text") or "No text."
                    dist = it.get("distance")
                    md = it.get("metadata") or {}
                    chunk_id = None
                    if md.get("doc_id") is not None and md.get("chunking_method") is not None and md.get("chunk_index") is not None:
                        chunk_id = f"{md['doc_id']}:{md['chunking_method']}:{md['chunk_index']}"

                    st.caption(f"Distance: {dist} | chunk_id: {chunk_id}")
                    st.markdown(
                        f"""
                        <div style="font-size:16px; line-height:1.55; padding:14px 16px; border-radius:12px;
                                    border:1px solid rgba(120,120,120,0.35); background:rgba(250,250,250,0.92);
                                    color:#000; white-space:pre-wrap; height:280px; overflow-y:auto;">
                        {txt}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # top1_box(c1, "Fixed (Token) Chunking — Top 1", fixed)
            # top1_box(c2, "Header (Markdown) Chunking — Top 1", header)

        # Raw debug JSON if you want to inspect further
        if dbg:
            with st.expander("Debug details"):
                st.json(dbg)

        st.session_state.chat_messages.append({"role": "assistant", "content": answer_text})
        return


def show_evaluation_page():
    import json
    import pandas as pd
    import plotly.express as px
    import requests
    import streamlit as st

    apply_theme("analytics")
    st.title("🧪 RAG Evaluation (Recall & MRR)")

    
    st.markdown(
        """
        <div class="eval-help">
        Upload an evaluation JSON file with this structure:

        ```json
        {
            "k": 5,
            "queries": [
            {
                "query": "...",
                "doc_id": "...",
                "relevant_chunk_ids": ["<doc_id>:<method>:<chunk_index>", "..."]
            }
            ]
        }
        ```
        </div>
        """,
        unsafe_allow_html=True
    )


    colA, colB = st.columns([2, 1])
    with colA:
        eval_file = st.file_uploader("Evaluation JSON", type=["json"])
    with colB:
        k = st.slider("Top-K (K)", min_value=1, max_value=20, value=5)

    if eval_file is None:
        st.info("Upload an evaluation JSON file to run evaluation.")
        return

    # Parse JSON safely
    try:
        payload = json.load(eval_file)
    except Exception as e:
        st.error(f"Invalid JSON: {e}")
        return

    if not isinstance(payload, dict):
        st.error("JSON root must be an object/dictionary.")
        return

    queries = payload.get("queries")
    if not isinstance(queries, list) or not queries:
        st.error("JSON must contain a non-empty `queries` list.")
        return

    # Basic schema validation
    required_fields = {"query", "doc_id", "relevant_chunk_ids"}
    bad_rows = []
    for idx, q in enumerate(queries):
        if not isinstance(q, dict):
            bad_rows.append((idx, "Each query item must be an object."))
            continue
        missing = required_fields - set(q.keys())
        if missing:
            bad_rows.append((idx, f"Missing fields: {sorted(list(missing))}"))
            continue
        if not isinstance(q.get("relevant_chunk_ids"), list) or len(q.get("relevant_chunk_ids")) == 0:
            bad_rows.append((idx, "`relevant_chunk_ids` must be a non-empty list."))

    if bad_rows:
        st.error("Evaluation JSON has invalid rows:")
        for idx, msg in bad_rows[:10]:
            st.write(f"- Row {idx}: {msg}")
        if len(bad_rows) > 10:
            st.write(f"...and {len(bad_rows) - 10} more.")
        return

    # Override K from UI (recommended so user can experiment)
    payload["k"] = int(k)

    # Show quick preview
    with st.expander("Preview loaded evaluation queries"):
        preview_df = pd.DataFrame(
            [{
                "query": q["query"],
                "doc_id": q["doc_id"],
                "n_relevant_chunk_ids": len(q.get("relevant_chunk_ids", []))
            } for q in queries]
        )
        st.dataframe(preview_df, use_container_width=True)

    if st.button("Run Evaluation"):
        try:
            resp = requests.post(f"{FASTAPI_URL}/eval/pdf", json=payload, timeout=300)
            if resp.status_code != 200:
                st.error(f"FastAPI error ({resp.status_code}): {resp.text}")
                return
            out = resp.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach FastAPI at {FASTAPI_URL}. Error: {e}")
            return
        except Exception as e:
            st.error(f"Unexpected error reading response: {e}")
            return

        summary = out.get("summary", {})
        details = out.get("details", [])
        k_used = out.get("k", k)

        if not summary:
            st.warning("No summary returned from backend.")
            return

        rows = []
        for method, vals in summary.items():
            rows.append({
                "method": method,
                f"Recall@{k_used}": float(vals.get("recall_at_k", 0.0)),
                f"MRR@{k_used}": float(vals.get("mrr_at_k", 0.0)),
                "n_queries": int(vals.get("n", 0)),
            })
        df_sum = pd.DataFrame(rows).sort_values("method")

        st.subheader("Summary")
        st.dataframe(df_sum, use_container_width=True)

        # --- Bar plots ---
        st.subheader("Comparison Plots")

        fig_recall = px.bar(df_sum, x="method", y=f"Recall@{k_used}", title=f"Recall@{k_used} by Chunking Method")
        st.plotly_chart(fig_recall, use_container_width=True)

        fig_mrr = px.bar(df_sum, x="method", y=f"MRR@{k_used}", title=f"MRR@{k_used} by Chunking Method")
        st.plotly_chart(fig_mrr, use_container_width=True)

        # --- Per-query breakdown ---
        if details:
            st.subheader("Per-query Breakdown")

            flat = []
            for d in details:
                qtext = d.get("query")
                doc_id = d.get("doc_id")
                pm = d.get("per_method", {}) or {}
                for method, mv in pm.items():
                    flat.append({
                        "query": qtext,
                        "doc_id": doc_id,
                        "method": method,
                        "recall": mv.get("recall"),
                        "mrr": mv.get("mrr"),
                    })

            df_det = pd.DataFrame(flat)
            st.dataframe(df_det, use_container_width=True)

            fig_mrr_dist = px.box(df_det, x="method", y="mrr", title="MRR distribution by Method")
            st.plotly_chart(fig_mrr_dist, use_container_width=True)

            fig_recall_dist = px.box(df_det, x="method", y="recall", title="Recall distribution by Method")
            st.plotly_chart(fig_recall_dist, use_container_width=True)

            # Optional: downloadable results
            st.subheader("Download Results")
            st.download_button(
                "Download summary as CSV",
                df_sum.to_csv(index=False).encode("utf-8"),
                file_name="rag_eval_summary.csv",
                mime="text/csv"
            )
            st.download_button(
                "Download per-query breakdown as CSV",
                df_det.to_csv(index=False).encode("utf-8"),
                file_name="rag_eval_per_query.csv",
                mime="text/csv"
            )
        else:
            st.info("No per-query details returned.")


def main():
    st.sidebar.header("Navigation")
    nav_options = ["📂 Upload Files", "📊 Analytics", "💬 ChatBot", "🧪 Evaluation"]
    selection = st.sidebar.radio("Select Page:", nav_options, label_visibility="collapsed")

    if selection == "📂 Upload Files":
        show_upload_page()
    elif selection == "📊 Analytics":
        show_analytics_page()
    elif selection == "💬 ChatBot":
        show_chatbot_page()
    elif selection == "🧪 Evaluation":
        show_evaluation_page()


if __name__ == "__main__":
    main()