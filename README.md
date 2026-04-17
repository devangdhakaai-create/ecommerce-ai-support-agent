# ecommerce-ai-support-agent
AI sales and customer support agent with RAG, memory, product Q&A, order lookup, and upsell workflow for e-commerce stores.

# 🛍️ E-Commerce Support Agent

AI chatbot for product Q&A, order help, upsell, and support.

## Stack
- **LLM**: LLaMA 3.3 70B via Groq API (free tier)
- **RAG**: FAISS + sentence-transformers (local, no API cost)
- **Memory**: Sliding window (last 6 turns)
- **UI**: Streamlit

## Setup

```bash
git clone <repo>
cd ecommerce_agent
pip install -r requirements.txt
```

`.env` file:

GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx

Run:
```bash
streamlit run app.py
```

## Features
- RAG on product catalog, FAQ, shipping & return policies
- Local embeddings — zero API cost for vector search
- Conversational memory across turns
- Sales-focused persona with upsell and bundle nudges
- Quick-action buttons + one-click chat reset