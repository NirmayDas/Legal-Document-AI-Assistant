from fastapi import FastAPI, Request
from pydantic import BaseModel
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from knowledge_graph import call_gemma, find_most_relevant_contract, RESULTS_JSON, EMBEDDINGS_PKL

app = FastAPI()

# Load contracts and embeddings at startup
with open(RESULTS_JSON, "r", encoding="utf-8") as f:
    contracts = json.load(f)
with open(EMBEDDINGS_PKL, "rb") as f:
    embeddings = pickle.load(f)
model = SentenceTransformer('all-MiniLM-L6-v2')

class AskRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask(request: AskRequest):
    query = request.query
    contract, score = find_most_relevant_contract(query, contracts, embeddings, model)
    context = json.dumps(contract, indent=2)
    prompt = f"Given the following contract information:\n{context}\n\nAnswer this question: {query}"
    response = call_gemma(prompt)
    return {"answer": response, "contract": contract, "score": float(score)}

@app.get("/")
def root():
    return {"message": "Contract Knowledge Graph API. Use POST /ask with {'query': 'your question'}"} 