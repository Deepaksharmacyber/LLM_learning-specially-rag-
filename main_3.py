from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
# from transformers.commands.serving import response_validator
import  requests
import json

def build_index(file_path="sample_text.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = [c.strip() for c in text.split(".") if c.strip()]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return model, index, chunks

def retrieve_chunks(query, model, index, chunks, k=2):
    query_emb = model.encode([query])
    distances, indices = index.search(np.array(query_emb), k)
    retrieved = [chunks[i] for i in indices[0]]
    return retrieved



def ask_llm(context,query):
    prompt = f"""
    You are helpful assistant.
    Use the following context to answer the question 
    
    Context :
    {context}
    
    Question:
    {query}
    
    Answers clearly and concisely based only on the context above
"""
    response = requests.post(
        "http://127.0.0.1:11434/v1/chat/completions",
        headers={"Content-Type":"application/json"},
        json={
            "model":"tinyllama", #on your llamafile model name
            "messages":[{"role":"user","content":prompt}]
        }
    )
    data = response.json()
    answer = data["choices"][0]["message"]["content"]
    return answer

def rag_pipeline(query):
    model,index ,chunks = build_index()
    retrieved = retrieve_chunks(query, model, index, chunks)
    context = "\n".join(retrieved)
    answer = ask_llm(context,query)

    print("🔍 Query:", query)
    print("\n📚 Retrieved Chunks:\n", context)
    print("\n🤖 LLM Answer:\n", answer)


if __name__ == "__main__":
    query = "what is natural language processing"
    rag_pipeline(query)
