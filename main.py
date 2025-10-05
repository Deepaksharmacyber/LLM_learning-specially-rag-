from sentence_transformers import SentenceTransformer, util

# Step 1: Create a small dataset (like a mini knowledge base)
sentences = [
    "I love playing football on weekends.",
    "Artificial intelligence is transforming the world.",
    "Python is a great programming language for beginners.",
    "Cooking pasta is easy and fun.",
    "The weather is sunny today.",
    "I enjoy reading about space exploration.",
    "Machine learning helps computers learn from data.",
    "Dogs are loyal and friendly animals.",
    "Meditation helps reduce stress and improve focus."
]

# Step 2: Load a small, fast model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 3: Convert sentences into embeddings (vectors)
sentence_embeddings = model.encode(sentences)

# Step 4: Take a user query
# query = "Which language is good for coding?"
query = "What weather is today"
query_embedding = model.encode(query)

# Step 5: Compute cosine similarity between query and all sentences
cosine_scores = util.cos_sim(query_embedding, sentence_embeddings)

# Step 6: Find the most similar sentence
best_match_index = cosine_scores.argmax()
print("🔍 Query:", query)
print("✅ Best match:", sentences[best_match_index])
print("📊 Similarity score:", cosine_scores[0][best_match_index].item())
