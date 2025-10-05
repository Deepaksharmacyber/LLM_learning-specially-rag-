# langchain_rag_fixed.py
"""
LangChain RAG demo — corrected for newer LangChain API.
- Uses sentence-transformers for embeddings via a small adapter that matches LangChain's Embeddings interface.
- Uses FAISS.from_texts(...) to build a vectorstore.
- Persists index locally and queries with a simple retriever + local Ollama LLM wrapper.
"""

import os
import json
import time
import requests
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer
from typing import List

# New community imports (avoid deprecated warnings)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# community vectorstore
from langchain_community.vectorstores import FAISS
# embeddings protocol adapter
from langchain.base_language import BaseLanguageModel  # (not used, but kept for clarity)
from langchain.embeddings.base import Embeddings

# ---------- Config ----------
PDF_PATH = "sample.pdf"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_DIR = "faiss_index"
LLM_API_URL = "http://127.0.0.1:11434/v1/chat/completions"
LLM_MODEL = "tinyllama"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3
# ----------------------------

# ---------- PDF -> LangChain Documents ----------
def load_pdf_to_docs(pdf_path: str) -> List[Document]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = " ".join(text.split())
            if text:
                pages.append((i, text))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    docs = []
    for (page_no, text) in pages:
        chunks = splitter.split_text(text)
        for idx, c in enumerate(chunks):
            docs.append(Document(page_content=c, metadata={"page": page_no, "chunk": idx}))
    return docs

# ---------- SentenceTransformers Embeddings adapter for LangChain ----------
class SentenceTransformersEmbeddings(Embeddings):
    """
    Implements the LangChain Embeddings interface using sentence-transformers.
    LangChain expects .embed_documents(List[str]) -> List[List[float]]
    and .embed_query(str) -> List[float]
    """
    def __init__(self, model_name=EMBED_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        arr = self.model.encode(texts, convert_to_numpy=True)
        return arr.astype(np.float32).tolist()

    def embed_query(self, text: str) -> List[float]:
        arr = self.model.encode([text], convert_to_numpy=True)[0]
        return arr.astype(np.float32).tolist()

# ---------- Minimal Ollama wrapper (same idea as earlier) ----------
class OllamaLLM:
    def __init__(self, api_url=LLM_API_URL, model=LLM_MODEL, timeout=(10,120)):
        self.api_url = api_url
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 256
        }
        resp = requests.post(self.api_url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

# ---------- Build / load FAISS vectorstore using LangChain helper ----------
def build_and_save_index(pdf_path=PDF_PATH, index_dir=INDEX_DIR):
    docs = load_pdf_to_docs(pdf_path)
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]
    print(f"Loaded {len(texts)} chunks from PDF.")

    embedder = SentenceTransformersEmbeddings()
    # LangChain's FAISS.from_texts expects an Embeddings instance
    vectorstore = FAISS.from_texts(texts, embedding=embedder, metadatas=metadatas)
    # persist
    os.makedirs(index_dir, exist_ok=True)
    vectorstore.save_local(index_dir)
    print("Index built and saved to", index_dir)
    return vectorstore

def load_index(index_dir=INDEX_DIR):
    embedder = SentenceTransformersEmbeddings()
    return FAISS.load_local(index_dir, embedder, allow_dangerous_deserialization=True)


# ---------- Query pipeline ----------
def run_rag_with_langchain(query: str):
    if not os.path.exists(INDEX_DIR):
        vs = build_and_save_index(PDF_PATH, INDEX_DIR)
    else:
        vs = load_index(INDEX_DIR)

    # Use 'similarity' search_type (supported). k = number of returned docs.
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})

    # Get documents
    docs = retriever.get_relevant_documents(query)
    print(f"Retrieved {len(docs)} docs:")
    for i, d in enumerate(docs, 1):
        print(i, "page:", d.metadata.get("page"), "chunk:", d.metadata.get("chunk"), "len:", len(d.page_content))
        print("--- excerpt ---")
        print(d.page_content[:300].strip(), "...\n")

    # Build context and call local LLM
    context = "\n\n".join([d.page_content for d in docs])
#     prompt = f"""Use ONLY the Context to answer the question. If you cannot find the answer, reply "I don't know".
#
# Context:
# {context}
#
# Question:
# {query}
#
# Answer concisely."""
    prompt = f'''
    You are a helpful assistant. RESPOND EXACTLY in this format and do NOT add anything else:

    "<QUOTE_FROM_CONTEXT>"
    Answer: <one concise sentence>

    If the answer is NOT present in the Context, reply exactly: "I don't know based on the provided document."

    Context:
    {context}

    Question:
    {query}
    '''

    llm = OllamaLLM()
    answer = llm.generate(prompt)
    print("\nAnswer:\n", answer)

if __name__ == "__main__":
    q = "What is natural language processing?"
    run_rag_with_langchain(q)
