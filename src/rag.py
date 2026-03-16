"""
rag.py — RAG query functions: embed question → search Pinecone → generate answer.
"""

import os
from typing import Optional, Tuple, List, Dict

from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "manual-rag")
EMBED_MODEL = "gemini-embedding-2-preview"  # Gemini multimodal embeddings
GEN_MODEL = "deepseek-chat"                 # DeepSeek for text generation
EMBED_DIM = 1536

# Lazy-initialized clients
_google_client: Optional[genai.Client] = None
_deepseek_client: Optional[OpenAI] = None
_pinecone_index = None


def _get_clients():
    global _google_client, _deepseek_client, _pinecone_index
    if _google_client is None:
        _google_client = genai.Client(api_key=GOOGLE_API_KEY)
    if _deepseek_client is None:
        _deepseek_client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
        )
    if _pinecone_index is None:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _pinecone_index = pc.Index(INDEX_NAME)
    return _google_client, _deepseek_client, _pinecone_index


def query_rag(question: str, top_k: int = 5) -> Tuple[str, List[Dict]]:
    """
    Full RAG pipeline:
    1. Embed the user question with gemini-embedding-2-preview (RETRIEVAL_QUERY)
    2. Search Pinecone for top_k similar page vectors
    3. Build context from matched pages
    4. Generate answer with DeepSeek
    5. Return (answer_text, sources_list)
    """
    google_client, deepseek_client, index = _get_clients()

    # 1. Embed question (Gemini multimodal embedding)
    embed_result = google_client.models.embed_content(
        model=EMBED_MODEL,
        contents=question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBED_DIM,
        ),
    )
    query_vector = embed_result.embeddings[0].values

    # 2. Search Pinecone
    search_result = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )

    matches = search_result.matches
    if not matches:
        return "I couldn't find relevant information in the manual.", []

    # 3. Build context from top matches
    context_parts = []
    for m in matches:
        page_num = m.metadata.get("page_number", "?")
        text = m.metadata.get("text", "")
        context_parts.append(f"[Page {page_num}]\n{text}")
    context = "\n\n---\n\n".join(context_parts)

    # 4. Generate answer with DeepSeek
    response = deepseek_client.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful Manual Assistant. Answer the user's question "
                    "based ONLY on the provided manual pages. Be concise and clear. "
                    "Use numbered steps when explaining procedures. "
                    "Mention relevant page numbers in your answer."
                ),
            },
            {
                "role": "user",
                "content": f"Manual content:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    answer = response.choices[0].message.content

    # 5. Build sources list
    sources = []
    for m in matches:
        sources.append({
            "page_number": int(m.metadata.get("page_number", 0)),
            "score": round(float(m.score) * 100, 1),
            "text_snippet": m.metadata.get("text", "")[:350],
            "image_path": m.metadata.get("image_path", ""),
        })

    return answer, sources
