# Streamlit UI: chat interface with session memory
# Streamlit UI: chat interface with session memory
import streamlit as st
from agent import get_response
from rag import load_knowledge_base, build_index

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="MyStore Support",
    page_icon="🛍️",
    layout="centered"
)

# ── Load RAG index once (cached so it doesn't reload every message) ──
@st.cache_resource
def init_rag():
    chunks = load_knowledge_base()
    index, _ = build_index(chunks)
    return chunks, index

chunks, index = init_rag()

# ── Session state for chat memory ──────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── UI Header ─────────────────────────────────────────────────
st.title("🛍️ MyStore AI Support")
st.caption("Ask about products, orders, returns, or get recommendations!")

# Quick prompt buttons for common queries
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("💻 Laptop help", key="btn_laptop"):
        st.session_state.prefill = "Tell me about your laptops"
with col2:
    if st.button("📦 Track order", key="btn_track"):
        st.session_state.prefill = "How do I track my order?"
with col3:
    if st.button("🎧 Recommend", key="btn_recommend"):
        st.session_state.prefill = "I need earbuds for the gym"

# ── Display chat history ───────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Handle prefill from quick buttons ─────────────────────────
prefill_text = st.session_state.pop("prefill", None)

# ── Chat input ────────────────────────────────────────────────
user_input = st.chat_input("Type your question...") or prefill_text

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Alex is typing..."):
            reply = get_response(
                user_input,
                st.session_state.messages[:-1],  # history excluding current
                chunks,
                index
            )
        st.markdown(reply)

    # Save assistant reply to memory
    st.session_state.messages.append({"role": "assistant", "content": reply})

# ── Sidebar: clear chat button ────────────────────────────────
with st.sidebar:
    st.header("🛒 MyStore")
    st.write("AI-powered support agent")
    st.divider()
    if st.button("🗑️ Clear chat", key="btn_clear"):
        st.session_state.messages = []
        st.rerun()
    st.caption("Powered by LLaMA 3.3 + FAISS RAG")