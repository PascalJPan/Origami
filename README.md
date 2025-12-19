# Origami - Mini Secondary Structure Predictor

Test it here: https://pascalpan.com/creations/origami/

Predict protein secondary structure (H/E/C) from an amino acid sequence.

- **Frontend:** React (Create React App), plain CSS (clean, a bit futuristic)
- **Backend:** FastAPI (Python) with a simple dummy model you can swap for a real one
- **Features:** up to 1000 AA, custom index start, number ruler every 10 residues, CSV export

---

## Demo Features

- Paste a sequence (A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y) — up to **1000** residues
- **Index Start** lets you offset residue numbering (default 1)
- Per‑residue colored grid (H=helix, E=sheet, C=coil) with tick marks every 10
- **Download CSV** containing `Index,Residue,State`
- API documented at **`/docs`** (FastAPI Swagger UI)

---

## Tech Stack

- **Frontend:** React 19 (CRA) + CSS
- **Backend:** FastAPI + Uvicorn
- **Python:** 3.8–3.11
- **Node:** 18 or 20 recommended

---

## Project Structure

```
.
├─ frontend/
│  ├─ public/
│  ├─ src/                  # React app source (App.js, App.css)
│  ├─ package.json
│  └─ .env.example
├─ backend/
│  ├─ main.py               # FastAPI app + /predict endpoint
│  ├─ model.py              # Model loading and inference
│  ├─ model_arch.py         # CNN architecture
│  ├─ requirements.txt
│  └─ weights/              # Model weights
├─ data/                    # Training data
├─ embeddings/              # Model embeddings
├─ esm_model.ipynb          # ESM model notebook
├─ modeling.ipynb           # Training notebook
└─ README.md
```

---

## Quick Start

### Option 1: Run Both Together (Easiest)

```bash
cd frontend
npm install
npm run dev
```

This starts both the backend (port 8000) and frontend (port 3000) with one command!

- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8000/docs

### Option 2: Run Separately (Two Terminals)

**Terminal 1 - Backend:**
```bash
cd backend
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

> **Windows:** use `.\.venv\Scripts\python` instead of `./.venv/bin/python`

**Terminal 2 - Frontend:**
```bash
cd frontend
npm install
npm start
```

The frontend uses a proxy to connect to the backend on port 8000. Click **Run prediction** → you should see a `POST /predict 200` in the backend terminal.

---
---
