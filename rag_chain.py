"""
utils/rag_chain.py
──────────────────
RAG chain that:
  - Answers STRICTLY from retrieved PDF chunks only
  - Returns both the answer AND source file references
  - No hallucination, no generic responses
"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import Document
from typing import List, Dict, Any

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "farm_plant_knowledge")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
QDRANT_HOST     = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_URL      = os.getenv("QDRANT_URL")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")

# ── Strict PDF-only prompt ────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an agricultural knowledge assistant. Answer ONLY using the context provided below, which is extracted from agricultural PDF documents.

STRICT RULES:
- Answer ONLY from the context below. Do NOT use outside knowledge.
- If the answer is not found in the context, respond with exactly: "This information is not available in the provided documents."
- Do NOT greet the user. Do NOT say "Hello", "Great question", "Fellow farmer", or any filler phrases.
- Be direct and factual. Start your answer immediately.
- Keep the answer concise and structured.

Context extracted from documents:
──────────────────────────────────
{context}
──────────────────────────────────

Question: {question}

Answer directly and only based on the context above:"""


# ── Embeddings ────────────────────────────────────────────────────────────────
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ── Retriever — returns docs with metadata ────────────────────────────────────
def get_retriever(embeddings: HuggingFaceEmbeddings, k: int = 5):
    qdrant_url = QDRANT_URL if QDRANT_URL else f"http://{QDRANT_HOST}:{QDRANT_PORT}"
    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        url=qdrant_url,
        api_key=QDRANT_API_KEY,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


# ── LLM ───────────────────────────────────────────────────────────────────────
def get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0.0,      # 0 = most factual, no creativity
        max_tokens=1024,
    )


# ── Format docs → context string + collect sources ───────────────────────────
def format_docs_with_sources(docs: List[Document]) -> Dict[str, Any]:
    """
    Returns:
      context : plain text of all chunks joined together
      sources : list of unique (file_name, page) tuples
    """
    context_parts = []
    sources = []

    for doc in docs:
        context_parts.append(doc.page_content)
        fname = doc.metadata.get("file_name", "Unknown file")
        page  = doc.metadata.get("page", "?")
        entry = {"file_name": fname, "page": page}
        if entry not in sources:
            sources.append(entry)

    return {
        "context": "\n\n".join(context_parts),
        "sources": sources,
    }


# ── Main chain builder ────────────────────────────────────────────────────────
def build_rag_chain():
    """
    Returns a callable that accepts a question string and returns:
    {
        "answer"  : str,
        "sources" : [{"file_name": ..., "page": ...}, ...]
    }
    """
    embeddings = get_embeddings()
    retriever  = get_retriever(embeddings)
    llm        = get_llm()

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=SYSTEM_PROMPT,
    )

    answer_chain = prompt | llm | StrOutputParser()

    def run(question: str) -> Dict[str, Any]:
        # 1. Retrieve relevant docs
        docs = retriever.invoke(question)

        # 2. Extract context text + sources
        parsed = format_docs_with_sources(docs)
        context = parsed["context"]
        sources = parsed["sources"]

        # 3. Generate answer strictly from context
        answer = answer_chain.invoke({
            "context": context,
            "question": question,
        })

        return {
            "answer": answer.strip(),
            "sources": sources,
        }

    return run