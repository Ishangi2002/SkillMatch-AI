import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# 1. Load processed job data
df = pd.read_csv("processed_jobs.csv")

# 2. Create job text (document for each job)
df["job_text"] = (
    "job title: " + df["Job Title"] +
    ", skills: " + df["Key Skills"] +
    ", role category: " + df["Role Category"] +
    ", industry: " + df["Industry"]
)

job_texts = df["job_text"].tolist()

# 3. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Generate embeddings for all jobs
job_embeddings = model.encode(job_texts, show_progress_bar=True)

# 5. Create FAISS index
embedding_dimension = job_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dimension)

# 6. Add embeddings to FAISS index
index.add(np.array(job_embeddings))

# 7. Save FAISS index to file
faiss.write_index(index, "faiss_index.bin")

# 8. Save job metadata (for retrieval later)
with open("job_metadata.pkl", "wb") as f:
    pickle.dump(df, f)

print("Vector store built successfully!")