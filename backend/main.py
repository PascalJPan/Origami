from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import re

from model import load_model, predict_secondary_structure

AA_RE = re.compile(r"[ACDEFGHIKLMNPQRSTVWYX]")

class PredictRequest(BaseModel):
    sequence: str = Field(..., min_length=1)
    index_start: int = Field(1, ge=1)

class PredictResponse(BaseModel):
    sequence: str
    index_start: int
    states: List[str]

app = FastAPI(title="Mini Secondary Structure API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

MODEL = None

@app.on_event("startup")
def _load():
    global MODEL
    MODEL = load_model()

@app.get("/healthz")
def healthz():
    return {"ok": True}

def clean_sequence(raw: str) -> str:
    up = (raw or "").upper()
    allowed = AA_RE.findall(up)
    return "".join(allowed)[:1000]  # cap at 1000

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    seq = clean_sequence(req.sequence)
    print("Cleaned sequence:", seq)  # Debug print
    if not seq:
        raise HTTPException(status_code=400, detail="No valid amino acids.")
    states = predict_secondary_structure(seq)
    print("Predicted states:", states)  # Debug print
    if len(states) != len(seq):
        raise HTTPException(status_code=500, detail="Model length mismatch.")
    return PredictResponse(sequence=seq, index_start=req.index_start, states=states)
