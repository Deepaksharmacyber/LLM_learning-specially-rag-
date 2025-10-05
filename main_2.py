from numpy.ma.core import indices
from sentence_transformers import  SentenceTransformer
import faiss
import numpy as np

from main import query_embedding

#step 1 : Load the text file
with open("sample_text.txt","r",encoding="utf-8") as f:
    text = f.read()

#step 2 : split into chunks (simple approach)
chunks = [chunk.strip() for chunk in text.split(".") if chunk.strip()]

print(f'number of chunks ,{len(chunks)}')
for i , c in enumerate(chunks,1):
    print(f'chunk {i} ,{c}')

#step 3 load model
model = SentenceTransformer("all-MiniLM-L6-v2")

#step 4 : create embeddings
embeddings = model.encode(chunks)

#step 5 : Initialize Faiss index
embeddings_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embeddings_dim) #L2 distance works fine

#step 6 : add embeddings to index
index.add(np.array(embeddings))
print(f'added , {index.ntotal} , chunks to Faiss index ')

#step 7 :  User query
query = "what is natural language processing"
query_embedding = model.encode([query])

#step 8 : search top-k results
k = 2  #number of most similar chunks to retrieve
distances , indices = index.search(np.array(query_embedding),k)

print(f'query {query} ')
print(f'Top results')
for i , idx in enumerate(indices[0]):
    print(f"{i + 1}. {chunks[idx]} (distance={distances[0][i]:.4f})")



