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
├─ backend/
│  ├─ main.py               # FastAPI app + /predict endpoint
│  ├─ model.py              # dummy predictor (replace with real model)
│  └─ requirements.txt
├─ public/
├─ src/                     # React app source (App.js, App.css)
├─ package.json
├─ .env.example
└─ README.md
```

---

## Quick Start (Two Terminals)

### 1) Backend (FastAPI)

```bash
cd backend
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
Open: http://127.0.0.1:8000/docs (try POST `/predict`)

> **Windows:** use `.\.venv\Scripts\python -m pip ...` and `.\.venv\Scripts\python -m uvicorn ...`

### 2) Frontend (React)

In a **new terminal** at repo root:
```bash
npm install

# Option A: CRA proxy (package.json already has it)
npm start

# Option B: env var (no proxy)
cp .env.example .env   # sets REACT_APP_API_BASE=http://127.0.0.1:8000
npm start
```
Click **Run prediction** → you should see a `POST /predict 200` in the backend terminal.

---
---
