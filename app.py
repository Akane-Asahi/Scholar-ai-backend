from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

app = FastAPI()

# CORS for frontend dev mode
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Quran data
with open("quran_data.json", "r", encoding="utf-8") as f:
    quran = json.load(f)

# Load or create embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

try:
    with open("embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
except:
    verses = [verse["text"] for verse in quran]
    embeddings = model.encode(verses, convert_to_tensor=True)
    with open("embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)

class Query(BaseModel):
    question: str

@app.post("/search")
def search(query: Query):
    input_vec = model.encode(query.question, convert_to_tensor=True)
    scores = torch.nn.functional.cosine_similarity(input_vec, embeddings)
    top_idx = torch.topk(scores, k=3).indices.tolist()

    results = [quran[i] for i in top_idx]
    return {"results": results}
