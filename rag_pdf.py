"""
RAG pipeline for PDF -> FAISS -> LLM (Ollama).
Features:
- PDF extraction (pdfplumber)
- Chunking with overlap
- Embeddings (sentence-transformers)
- FAISS index build & save/load
- Retrieval, dedupe
- Extract supporting sentence(s) heuristic
- Robust LLM call with longer timeouts & retries (Ollama default port 11434)
- Strict prompt: QUOTE supporting sentence, then "Answer: <one-sentence>"
- Fallback to returning retrieved sentence/chunk if LLM not available or unreliable
"""

from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np
import pdfplumber
import pickle
import requests
import json
import os
import sys
import time
import hashlib
from typing import List, Tuple
from nltk.tokenize import sent_tokenize

# ---------- Config ----------
PDF_PATH = "AI_ML_NLP_Overview.pdf"            # change to your PDF filename
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_URL = "http://127.0.0.1:11434/v1/chat/completions"  # Ollama default
LLM_MODEL = "tinyllama"            # adjust if necessary
CHUNK_SIZE = 1000                  # characters per chunk
CHUNK_OVERLAP = 200                # overlap chars
TOP_K = 3
INDEX_PREFIX = "pdf_index"
# ----------------------------

# ---------- PDF extraction ----------
def extract_text_from_pdf(pdf_path: str) -> List[Tuple[int, str]]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = " ".join(text.split())
            pages.append((i, text))
    return pages

# ---------- Chunking ----------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += step
    return chunks

def build_chunks_from_pdf(pdf_path: str):
    pages = extract_text_from_pdf(pdf_path)
    all_chunks = []
    metadatas = []
    for page_no, page_text in pages:
        page_chunks = chunk_text(page_text)
        for idx, ch in enumerate(page_chunks):
            all_chunks.append(ch)
            metadatas.append({
                "pdf": os.path.basename(pdf_path),
                "page": page_no,
                "chunk_id": idx
            })
    return all_chunks, metadatas

# ---------- FAISS & embeddings ----------
def build_faiss_index(chunks: List[str], model_name=EMBED_MODEL_NAME):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return model, index, embeddings

def save_index(index, embeddings, metadatas, chunks, prefix=INDEX_PREFIX):
    faiss.write_index(index, prefix + ".faiss")
    np.save(prefix + "_embeddings.npy", embeddings)
    with open(prefix + "_meta.pkl", "wb") as f:
        pickle.dump(metadatas, f)
    with open(prefix + "_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

def load_index(prefix=INDEX_PREFIX):
    index = faiss.read_index(prefix + ".faiss")
    embeddings = np.load(prefix + "_embeddings.npy")
    with open(prefix + "_meta.pkl", "rb") as f:
        metadatas = pickle.load(f)
    with open(prefix + "_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, embeddings, metadatas, chunks

# ---------- Retrieval ----------
def retrieve(query: str, model: SentenceTransformer, index, chunks: List[str], embeddings, k=TOP_K):
    q_emb = model.encode([query], convert_to_numpy=True)
    if q_emb.dtype != np.float32:
        q_emb = q_emb.astype(np.float32)
    distances, indices = index.search(q_emb, k)
    indices = indices[0]
    distances = distances[0]
    candidate_embs = embeddings[indices]
    cos_scores = util.cos_sim(q_emb, candidate_embs)[0].cpu().numpy()
    results = []
    for i, idx in enumerate(indices):
        results.append({
            "index": int(idx),
            "distance": float(distances[i]),
            "cosine": float(cos_scores[i]),
            "text": chunks[idx]
        })
    # sort by cosine descending (semantic similarity)
    results = sorted(results, key=lambda r: r["cosine"], reverse=True)
    return results

# ---------- Helpers: dedupe, extract supporting sentences ----------
def dedupe_results(results):
    unique = []
    seen = set()
    for r in results:
        if r["index"] not in seen:
            seen.add(r["index"])
            unique.append(r)
    return unique

# def extract_supporting_sentences(chunk_text: str, query: str) -> List[str]:
#     q = query.lower()
#     keywords = []
#     if "natural language" in q or "nlp" in q:
#         keywords = ["natural language", "nlp", "language processing"]
#     else:
#         keywords = [w for w in q.split() if len(w) >= 4]
#     sentences = [s.strip() for s in chunk_text.split('.') if s.strip()]
#     matched = []
#     for s in sentences:
#         ls = s.lower()
#         for kw in keywords:
#             if kw in ls:
#                 # add with trailing period
#                 matched.append(s.strip() + ".")
#                 break
#     return matched

# ---------- Robust LLM call (with retries and long read timeout) ----------

def extract_supporting_sentences(chunk_text: str, query: str, max_sents=2) -> List[str]:
    """
    Use nltk.sent_tokenize to split into real sentences, then return up to max_sents
    that contain query keywords. Result is deduped (preserve order).
    """
    q = query.lower()
    # build keyword list (can be extended)
    if "natural language" in q or "nlp" in q:
        keywords = ["natural language", "nlp", "language processing"]
    else:
        tokens = [w for w in q.split() if len(w) >= 4]
        keywords = tokens if tokens else [q]

    # robust sentence splitting
    sentences = sent_tokenize(chunk_text)
    matched = []
    seen = set()
    for s in sentences:
        ls = s.lower()
        for kw in keywords:
            if kw in ls:
                s_clean = s.strip()
                if s_clean not in seen:
                    matched.append(s_clean if s_clean.endswith('.') else s_clean + '.')
                    seen.add(s_clean)
                break
        if len(matched) >= max_sents:
            break
    return matched


def call_llm(payload, url=LLM_URL, timeout_connect=10, timeout_read=120, max_retries=1):
    attempt = 0
    while True:
        attempt += 1
        try:
            start = time.time()
            resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=(timeout_connect, timeout_read))
            elapsed = time.time() - start
        except requests.exceptions.ReadTimeout:
            if attempt <= max_retries:
                print(f"[LLM] ReadTimeout on attempt {attempt}. Retrying...")
                continue
            raise RuntimeError(f"ReadTimeout after {timeout_read}s contacting {url}")
        except requests.exceptions.ConnectTimeout:
            raise RuntimeError(f"ConnectTimeout ({timeout_connect}s) when connecting to {url}")
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Connection error contacting {url}: {e}")

        if not resp.ok:
            body = resp.text
            raise RuntimeError(f"Server returned status {resp.status_code} after {elapsed:.1f}s. Body:\n{body[:2000]}")
        try:
            data = resp.json()
        except Exception:
            raise RuntimeError(f"Response not JSON (status {resp.status_code}). Text:\n{resp.text[:2000]}")

        # parse common OpenAI-compatible shapes
        try:
            choice0 = data["choices"][0]
            if "message" in choice0 and "content" in choice0["message"]:
                return choice0["message"]["content"]
            elif "text" in choice0:
                return choice0["text"]
            else:
                return json.dumps(choice0, ensure_ascii=False)
        except Exception:
            return json.dumps(data, ensure_ascii=False)

# ---------- High-level ask with extracted context and strict prompt ----------
def ask_llm_with_context_strict(retrieved_results, query,
                                url=LLM_URL, model_name=LLM_MODEL,
                                timeout_connect=10, timeout_read=120):
    # dedupe retrieved chunks first
    results = dedupe_results(retrieved_results)
    if len(results) == 0:
        return None

    # Try to extract up to 2 supporting sentences (but we'll prefer 1)
    support_sents = []
    for r in results:
        sents = extract_supporting_sentences(r["text"], query, max_sents=2)
        for s in sents:
            if s not in support_sents:
                support_sents.append(s)
        if len(support_sents) >= 2:
            break

    # Dedup already handled; choose best context
    if support_sents:
        # use only the top 1 supporting sentence (most precise)
        context = support_sents[0]
        print("Using SINGLE supporting sentence as context:")
        print(" -", context)
    else:
        # fallback: use top unique chunk (compact)
        context = results[0]["text"]
        print("No direct supporting sentence found. Using top chunk as context (compact).")

    # Strict prompt: QUOTE only + Answer: one short sentence. Nothing else.
    prompt = f"""
You are a helpful assistant. RESPOND EXACTLY in the following format and DO NOT add any extra text:

"<QUOTE_FROM_CONTEXT>"
Answer: <one concise sentence that answers the question>

If the answer is NOT present in the Context, reply exactly:
"I don't know based on the provided document."

Context:
{context}

Question:
{query}
"""
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "Follow the user's output format exactly. No extra text."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 80,
        "top_p": 1.0
    }

    try:
        res = call_llm(payload, url=url, timeout_connect=timeout_connect, timeout_read=timeout_read)
        # quick post-processing: strip leading/trailing whitespace
        if isinstance(res, str):
            return res.strip()
        return res
    except Exception as e:
        print("[LLM] call failed or returned error:", e)
        # fallback: return the supporting sentence verbatim (no LLM)
        if support_sents:
            return f'"{support_sents[0]}"\nAnswer: {support_sents[0]}'
        else:
            return f'"{context}"\nAnswer: {context[:200]}'

# ---------- Full pipeline query ----------
def rag_query_pdf(query: str, prefix=INDEX_PREFIX, k=TOP_K):
    # load index & resources
    index, embeddings, metadatas, chunks = load_index(prefix)
    model = SentenceTransformer(EMBED_MODEL_NAME)
    results = retrieve(query, model, index, chunks, embeddings, k=k)

    # attach metadata
    for r in results:
        r["metadata"] = metadatas[r["index"]]

    # print retrieved items
    print("Retrieved (sorted by cosine):")
    for i, r in enumerate(results, 1):
        print(f"{i}. page={r['metadata']['page']} chunk_id={r['metadata']['chunk_id']} cosine={r['cosine']:.4f}")
        print(r["text"][:400].strip(), "...\n")

    # dedupe & ask LLM with strict prompt
    answer = ask_llm_with_context_strict(results, query)
    if answer:
        print("\nLLM answer:\n", answer)
        return answer, results

    # fallback: try to return a concise supporting sentence if available
    # extract from top result
    top = dedupe_results(results)[0] if results else None
    if top:
        sents = extract_supporting_sentences(top["text"], query)
        if sents:
            print("\nFallback (no LLM): returning supporting sentence verbatim:")
            print(sents[0])
            return sents[0], results
        else:
            print("\nFallback (no LLM): returning top chunk verbatim:")
            print(top["text"])
            return top["text"], results

    # nothing found
    print("No results found.")
    return None, results

# ---------- Build index helper ----------
def build_and_save_index_from_pdf(pdf_path: str, prefix=INDEX_PREFIX):
    print("Extracting + chunking PDF...")
    chunks, metadatas = build_chunks_from_pdf(pdf_path)
    print(f"Total chunks: {len(chunks)}")
    if len(chunks) == 0:
        raise RuntimeError("No text chunks extracted from PDF. Is the PDF text-based or scanned?")
    print("Building embeddings & FAISS index (this may take a moment)...")
    model, index, embeddings = build_faiss_index(chunks)
    print("Saving index and metadata...")
    save_index(index, embeddings, metadatas, chunks, prefix)
    print("Done. Index saved.")
    return model, index, chunks, embeddings, metadatas

def file_sha256(path,block_size=65536):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        while True:
            data = f.read(block_size)
            if not data:
                break
            h.update(data)
    return h.hexdigest()

# ---------- CLI / main ----------
if __name__ == "__main__":
    # CLI: optional PDF path & optional query
    pdf_path = PDF_PATH
    if len(sys.argv) >= 2:
        pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"PDF not found: {pdf_path}")
        sys.exit(1)

    # compute current pdf hash
    pdf_hash = file_sha256(pdf_path)
    meta_hash_file = INDEX_PREFIX + ".pdfhash"

    need_build = False
    # if index files missing -> build
    if not os.path.exists(INDEX_PREFIX + ".faiss") or not os.path.exists(INDEX_PREFIX + "_embeddings.npy"):
        print("Index files not found — will build index.")
        need_build = True
    else:
        # if hash file missing -> build (safer)
        if not os.path.exists(meta_hash_file):
            print("No saved PDF hash found — will build index.")
            need_build = True
        else:
            saved_hash = open(meta_hash_file, "r").read().strip()
            if saved_hash != pdf_hash:
                print("PDF has changed since last index build — rebuilding index.")
                need_build = True
            else:
                print("PDF unchanged; using existing index.")

    if need_build:
        # remove old index files (safe cleanup)
        for fname in [INDEX_PREFIX + ".faiss", INDEX_PREFIX + "_embeddings.npy", INDEX_PREFIX + "_meta.pkl", INDEX_PREFIX + "_chunks.pkl"]:
            if os.path.exists(fname):
                os.remove(fname)
        # build & save index
        build_and_save_index_from_pdf(pdf_path, prefix=INDEX_PREFIX)
        # save the pdf hash
        with open(meta_hash_file, "w") as f:
            f.write(pdf_hash)

    # Query: from CLI if passed, else default
    query = "Three main types of machine learning"
    if len(sys.argv) >= 3:
        query = " ".join(sys.argv[2:])

    print(f"\nQuery: {query}\n")
    answer, retrieved = rag_query_pdf(query, prefix=INDEX_PREFIX, k=TOP_K)
