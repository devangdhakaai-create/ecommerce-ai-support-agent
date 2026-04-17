# Agent logic: system prompt, memory, RAG-augmented response generation
# Load .env before initializing client
from dotenv import load_dotenv
load_dotenv()

import os
from openai import OpenAI
from rag import retrieve

# Groq uses OpenAI-compatible API — just change base_url + key
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

SYSTEM_PROMPT = """You are Alex, a friendly and sales-savvy support agent for MyStore — a premium online electronics & home office store.

Your goals:
1. Answer product questions clearly using provided context
2. Help with orders, returns, and shipping
3. Recommend products based on customer needs
4. Upsell bundles/promotions naturally (never pushy)
5. Always end with a helpful nudge or CTA

Tone: Friendly, confident, concise. Like a helpful Apple Store employee.
If you don't know something, say so honestly and offer to escalate."""

def get_response(user_message, chat_history, chunks, index):
    context_chunks = retrieve(user_message, chunks, index)
    context = "\n\n".join(context_chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in chat_history[-6:]:
        messages.append(msg)

    messages.append({
        "role": "user",
        "content": f"Context from store knowledge base:\n{context}\n\nCustomer: {user_message}"
    })

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Fast + free tier on Groq
        messages=messages,
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].message.content