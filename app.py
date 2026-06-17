import uuid
from datetime import datetime

import streamlit as st
from groq import Groq
from dotenv import load_dotenv

from ingest import chunk_uploaded_pdf
from embeddings import get_or_create_collection, add_chunks_to_collection, delete_chat_collection
from retriever import search

load_dotenv()
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]


def build_messages(system_prompt, history, context, question):
    messages = [{"role": "system", "content": system_prompt}]
    for entry in history:
        messages.append({"role": "user", "content": entry["question"]})
        messages.append({"role": "assistant", "content": entry["answer"]})
    messages.append(
        {
            "role": "user",
            "content": f"Context from documents:\n{context}\n\nQuestion: {question}",
        }
    )
    return messages


def generate_answer(question, context_chunks, history):
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    )
    system_prompt = (
        "You are a helpful study assistant. Answer questions using ONLY "
        "the provided context. Cite which source and page you got info "
        "from. If the context doesn't contain the answer, say so clearly. "
        "Consider previous conversation turns for context on follow-up questions."
    )
    client = Groq()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=build_messages(system_prompt, history, context, question),
    )
    return response.choices[0].message.content


def new_chat():
    chat_id = str(uuid.uuid4())[:8]
    st.session_state.chats[chat_id] = {
        "title": "New chat",
        "created": datetime.now().isoformat(),
        "history": [],
        "uploaded_files": [],
    }
    st.session_state.active_chat = chat_id
    st.session_state.show_settings = False


def delete_chat(chat_id):
    delete_chat_collection(chat_id)
    if chat_id in st.session_state.chats:
        del st.session_state.chats[chat_id]
    if st.session_state.active_chat == chat_id:
        if st.session_state.chats:
            st.session_state.active_chat = list(st.session_state.chats.keys())[0]
        else:
            new_chat()


st.set_page_config(
    page_title="Study Notes Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.25rem;
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
    }
    .source-card {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.85rem;
        margin: 0.5rem 0;
    }
    .source-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #3b82f6;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
    }
    .uploaded-file {
        font-size: 0.85rem;
        color: #4b5563;
        padding: 4px 0;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
    }
    section[data-testid="stSidebar"] hr {
        margin: 0.75rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "chats" not in st.session_state:
    st.session_state.chats = {}
if "active_chat" not in st.session_state:
    st.session_state.active_chat = None
if "show_settings" not in st.session_state:
    st.session_state.show_settings = False
if "n_results" not in st.session_state:
    st.session_state.n_results = 5

if not st.session_state.chats:
    new_chat()


with st.sidebar:
    st.markdown("### 📚 Study Notes Q&A")

    if st.button("➕ New chat", use_container_width=True, type="primary"):
        new_chat()
        st.rerun()

    st.markdown("---")
    st.markdown("**Your chats**")

    for chat_id, chat in list(st.session_state.chats.items()):
        is_active = chat_id == st.session_state.active_chat
        col_chat, col_del = st.columns([5, 1])

        with col_chat:
            label = chat["title"] if chat["title"] else "New chat"
            if len(label) > 28:
                label = label[:28] + "…"
            if st.button(
                label,
                key=f"select_{chat_id}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.active_chat = chat_id
                st.session_state.show_settings = False
                st.rerun()

        with col_del:
            if st.button("✕", key=f"del_{chat_id}", help="Delete this chat"):
                delete_chat(chat_id)
                st.rerun()

    st.markdown("---")

    if st.button("⚙️ Settings", use_container_width=True):
        st.session_state.show_settings = not st.session_state.show_settings
        st.rerun()

    if st.session_state.show_settings:
        st.markdown("**Sources per answer**")
        st.session_state.n_results = st.slider(
            "Sources per answer",
            min_value=1,
            max_value=10,
            value=st.session_state.n_results,
            label_visibility="collapsed",
        )


active_id = st.session_state.active_chat
active_chat = st.session_state.chats[active_id]
collection = get_or_create_collection(active_id)

st.markdown(
    f'<div class="main-header">{active_chat["title"]}</div>',
    unsafe_allow_html=True,
)

with st.expander(
    f"📎 Documents ({len(active_chat['uploaded_files'])})",
    expanded=len(active_chat["uploaded_files"]) == 0,
):
    uploaded = st.file_uploader(
        "Upload PDFs to this chat",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{active_id}",
    )

    if uploaded:
        new_files = [
            f for f in uploaded if f.name not in active_chat["uploaded_files"]
        ]
        if new_files:
            with st.spinner(f"Processing {len(new_files)} file(s)..."):
                for f in new_files:
                    chunks = chunk_uploaded_pdf(f)
                    doc_id = f"{f.name}_{uuid.uuid4().hex[:6]}"
                    add_chunks_to_collection(collection, chunks, doc_id)
                    active_chat["uploaded_files"].append(f.name)
            st.success(f"Added {len(new_files)} document(s)")
            st.rerun()

    if active_chat["uploaded_files"]:
        st.markdown("**In this chat:**")
        for fname in active_chat["uploaded_files"]:
            st.markdown(
                f'<div class="uploaded-file">📄 {fname}</div>',
                unsafe_allow_html=True,
            )


for entry in active_chat["history"]:
    with st.chat_message("user"):
        st.markdown(entry["question"])
    with st.chat_message("assistant"):
        st.markdown(entry["answer"])
        if entry["sources"]:
            with st.expander(f"📎 Show sources ({len(entry['sources'])})"):
                for c in entry["sources"]:
                    st.markdown(
                        f"""
                        <div class="source-card">
                            <div class="source-label">{c['source']}</div>
                            <div style="color: #4b5563; font-size: 0.88rem;">
                                {c['text'][:400]}...
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

if not active_chat["uploaded_files"]:
    st.info("👆 Upload one or more PDFs above to start asking questions.")
else:
    question = st.chat_input("Ask a question about your documents...")
    if question:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching and thinking..."):
                chunks = search(
                    collection, question, n_results=st.session_state.n_results
                )
                answer = generate_answer(
                    question, chunks, active_chat["history"]
                )

            st.markdown(answer)
            if chunks:
                with st.expander(f"📎 Show sources ({len(chunks)})"):
                    for c in chunks:
                        st.markdown(
                            f"""
                            <div class="source-card">
                                <div class="source-label">{c['source']}</div>
                                <div style="color: #4b5563; font-size: 0.88rem;">
                                    {c['text'][:400]}...
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        active_chat["history"].append(
            {"question": question, "answer": answer, "sources": chunks}
        )

        if active_chat["title"] == "New chat":
            active_chat["title"] = question[:40] + ("…" if len(question) > 40 else "")
            st.rerun()