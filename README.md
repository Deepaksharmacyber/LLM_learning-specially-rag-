📘 RAG Learning Journey (Retrieval-Augmented Generation)

This repository documents my step-by-step learning journey of building a Retrieval-Augmented Generation (RAG) system from scratch — from understanding embeddings to integrating a language model and working with PDFs.

🚀 Phases Overview
Phase 1 — Foundations: Semantic Search Demo

File: 01_semantic_search_demo.py

Store 5–10 sentences in a list.

Generate embeddings using SentenceTransformers.

Compute cosine similarity between a user query and each sentence.

Return the most semantically similar sentence.
🎯 Goal: Understand embeddings and cosine similarity.

Phase 2 — Mini RAG (Without LLM)

File: 02_faiss_retrieval_demo.py

Split a small text file into chunks.

Convert chunks into embeddings and store in a FAISS index.

Query → Retrieve most relevant chunks based on similarity.
🎯 Goal: Understand retrieval and FAISS indexing.

Phase 3 — RAG with LLM

File: 03_rag_with_llm.py

Add a small LLM (like llamafile API).

Combine retrieved chunks and user query into a prompt.

LLM generates an answer grounded in the retrieved content.
🎯 Goal: Complete the Retrieval-Augmented Generation loop.

Phase 4 — PDF Support

File: 04_pdf_rag_system.py

Replace text file with a PDF extractor (e.g., pdfplumber).

Chunk and embed PDF text, store embeddings in FAISS.

Ask questions about PDF content.
🎯 Goal: Build a working “Ask your PDF” system.

Phase 5 — Framework Integration (LangChain / LlamaIndex)

File: 05_langchain_rag_pipeline.py

Recreate the same pipeline using LangChain or LlamaIndex.

Compare manual and framework-based implementations.
🎯 Goal: Learn how frameworks simplify RAG development.

🧩 Tech Stack

Python 3.x

SentenceTransformers – for embeddings

FAISS – for similarity search

pdfplumber – for PDF text extraction

Llama API / llamafile – for LLM responses

LangChain / LlamaIndex – for structured RAG pipeline