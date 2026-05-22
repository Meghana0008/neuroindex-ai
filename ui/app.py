import json

import requests
import streamlit as st

API = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="NeuroIndex AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.title("🧠 NeuroIndex AI")
    st.caption("Multimodal RAG · Page-Level Citations · Graph Intelligence")
    st.divider()

    st.subheader("Agent")
    agent_type = st.selectbox(
        "Select agent",
        ["general", "financial", "legal", "research", "hr"],
        format_func=lambda x: {
            "general":   "🔍 General",
            "financial": "💰 Financial Analyst",
            "legal":     "⚖️ Legal Analyst",
            "research":  "🔬 Research Analyst",
            "hr":        "👥 HR Analyst",
        }[x],
        label_visibility="collapsed",
    )

    st.divider()

    st.subheader("Retrieval Options")
    use_graph = st.toggle("Graph Retrieval", value=True)
    use_hyde = st.toggle("HyDE Expansion", value=True)
    use_multi_query = st.toggle("Multi-Query", value=True)

    st.divider()

    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader("Choose PDFs", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        if st.button("Index Documents", type="primary", use_container_width=True):
            for uploaded_file in uploaded_files:
                with st.spinner(f"Indexing '{uploaded_file.name}'…"):
                    try:
                        resp = requests.post(
                            f"{API}/upload",
                            files={"file": (uploaded_file.name, uploaded_file, "application/pdf")},
                            timeout=600,
                        )
                        if resp.status_code == 200:
                            d = resp.json()
                            st.success(
                                f"✅ **{d['filename']}** — "
                                f"Pages: **{d['num_pages']}** | "
                                f"Chunks: **{d['num_child_chunks']}**"
                            )
                        else:
                            st.error(f"Failed '{uploaded_file.name}': {resp.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot reach API. Is `uvicorn api.main:app` running?")
                    except Exception as e:
                        st.error(str(e))

    st.divider()

    st.subheader("Index Stats")
    try:
        health = requests.get(f"{API}/health", timeout=3).json()
        col1, col2 = st.columns(2)
        col1.metric("Chunks", health.get("indexed_chunks", 0))
        col2.metric("Vectors", health.get("faiss_vectors", 0))
        col1.metric("KG Nodes", health.get("graph_nodes", 0))
        col2.metric("KG Edges", health.get("graph_edges", 0))
    except Exception:
        st.warning("⚠ API not reachable")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────
st.header("Ask Your Documents")
st.caption("Hybrid Dense + Sparse + Graph retrieval → BGE Reranking → GPT-4o with page-level citations")

if "messages" not in st.session_state:
    st.session_state.messages = []

# history sent to LLM — only role+content, last 6 messages
if "history" not in st.session_state:
    st.session_state.history = []

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("citations"):
            with st.expander(f"📄 {len(msg['citations'])} source(s)"):
                for c in msg["citations"]:
                    st.markdown(f"**{c['doc_name']}** — Page **{c['page_number']}**")
                    if c.get("chunk_content"):
                        st.caption(c["chunk_content"][:400] + "…")
                    st.divider()

        if msg.get("query_variants") and len(msg["query_variants"]) > 1:
            with st.expander("🔍 Query variants used"):
                for v in msg["query_variants"]:
                    st.markdown(f"- {v}")

        if msg.get("retrieved_chunks"):
            with st.expander("🗂 Retrieved chunks (debug)"):
                for chunk in msg["retrieved_chunks"]:
                    st.markdown(
                        f"**{chunk['doc_name']}** — Page {chunk['page_number']} "
                        f"| rerank score: `{chunk['rerank_score']:.4f}`"
                    )
                    st.caption(chunk["content"])
                    st.divider()

# Chat input
if user_query := st.chat_input("Ask a question about your documents…"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""
        meta = {}

        payload = {
            "query": user_query,
            "use_graph": use_graph,
            "use_hyde": use_hyde,
            "use_multi_query": use_multi_query,
            "conversation_history": st.session_state.history,
            "agent_type": agent_type,
        }

        try:
            with requests.post(f"{API}/query/stream", json=payload, stream=True, timeout=180) as r:
                if r.status_code != 200:
                    st.error(f"Error {r.status_code}: {r.text}")
                else:
                    for line in r.iter_lines():
                        if not line:
                            continue
                        decoded = line.decode("utf-8")
                        if not decoded.startswith("data: "):
                            continue
                        content = decoded[6:]
                        if content == "[DONE]":
                            break
                        try:
                            chunk = json.loads(content)
                        except Exception:
                            continue

                        if chunk["type"] == "token":
                            full_answer += chunk["content"]
                            placeholder.markdown(full_answer + "▌")
                        elif chunk["type"] == "done":
                            meta = chunk
                            full_answer = chunk.get("answer", full_answer)
                        elif chunk["type"] == "error":
                            st.error(chunk["content"])

            placeholder.markdown(full_answer)

            if meta.get("processing_time"):
                st.caption(f"⏱ {meta['processing_time']}s")

            if meta.get("citations"):
                with st.expander(f"📄 {len(meta['citations'])} source(s)"):
                    for c in meta["citations"]:
                        st.markdown(f"**{c['doc_name']}** — Page **{c['page_number']}**")
                        if c.get("chunk_content"):
                            st.caption(c["chunk_content"][:400] + "…")
                        st.divider()

            if len(meta.get("query_variants", [])) > 1:
                with st.expander("🔍 Query variants used"):
                    for v in meta["query_variants"]:
                        st.markdown(f"- {v}")

            if meta.get("retrieved_chunks"):
                with st.expander("🗂 Retrieved chunks (debug)"):
                    for chunk in meta["retrieved_chunks"]:
                        st.markdown(
                            f"**{chunk['doc_name']}** — Page {chunk['page_number']} "
                            f"| rerank score: `{chunk['rerank_score']:.4f}`"
                        )
                        st.caption(chunk["content"])
                        st.divider()

            # Save to display history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "citations": meta.get("citations", []),
                "query_variants": meta.get("query_variants", []),
                "retrieved_chunks": meta.get("retrieved_chunks", []),
            })

            # Update LLM context window — keep last 6 messages (3 turns)
            st.session_state.history.append({"role": "user", "content": user_query})
            st.session_state.history.append({"role": "assistant", "content": full_answer})
            st.session_state.history = st.session_state.history[-6:]

        except requests.exceptions.ConnectionError:
            st.error("Cannot reach API. Make sure `uvicorn api.main:app` is running.")
        except Exception as e:
            st.error(str(e))
