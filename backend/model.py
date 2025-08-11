from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import json
import numpy as np
import torch

# --- your architecture ---
from model_arch import ProteinClassifier2  # expects input [B, L, 24] → logits [B, L, C]

# ---------- paths ----------
WEIGHTS_DIR = Path(__file__).parent / "weights"
TS_PATH = WEIGHTS_DIR / "model_weights.ts"      # optional TorchScript artifact
CKPT_PATH = WEIGHTS_DIR / "model_weights.pt"    # state_dict checkpoint
META_PATH = WEIGHTS_DIR / "meta.json"          # metadata saved during training

# ---------- AA + labels ----------
DEFAULT_LABELS = ["H", "E", "C"]  # Q3
LABELS = DEFAULT_LABELS

# ---------- globals ----------
_MODEL_TS = None          # torch.jit.ScriptModule
_MODEL = None             # torch.nn.Module
_META: Dict[str, Any] = {}
NUM_CLASSES = 3

# ---------- load meta ----------
if META_PATH.exists():
    with open(META_PATH) as f:
        _META = json.load(f)
    NUM_CLASSES = int(_META.get("num_classes", 3))
    # use label map from meta if provided
    if "label_map" in _META:
        # meta may store keys as strings; ensure index order 0..C-1
        label_map = _META["label_map"]
        try:
            LABELS = [label_map[str(i)] for i in range(NUM_CLASSES)]
        except Exception:
            LABELS = DEFAULT_LABELS
else:
    _META = {"arch": "ProteinClassifier2", "num_classes": NUM_CLASSES}

# ---------- feature pipeline (REPLACE with your real 24-dim features) ----------
def _one_hot_20(seq: str) -> np.ndarray:
    L = len(seq)
    arr = np.zeros((L, 20), dtype=np.float32)
    for i, a in enumerate(seq):
        j = AA_TO_IDX.get(a)
        if j is not None:
            arr[i, j] = 1.0
    return arr  # [L,20]

def _extra_4_features(seq: str) -> np.ndarray:
    L = len(seq)
    return np.zeros((L, 4), dtype=np.float32)

# 22 AA codes (20 canonicals + X + '-'), same order as in training
AA_CODES = list("ACDEFGHIKLMNPQRSTVWY") + ["X"] + ["-"]
AA_TO_IDX = {a: i for i, a in enumerate(AA_CODES)}  # 0..21

def _featurize(seq: str) -> torch.Tensor:
    """
    Build per-residue features as used in training:
      - 22-d one-hot over AA_CODES (20 + X + '-')
      - 2-d terminal flags: [is_N_terminal, is_C_terminal]
    Output tensor shape: [1, L, 24] (float32)
    Notes:
      - user inputs won’t contain '-', so that column will be 0s
      - unknown letters (if any) become 'X'
    """
    seq = seq.upper()
    L = len(seq)

    # one-hot 22
    onehot = np.zeros((L, 22), dtype=np.float32)
    for i, ch in enumerate(seq):
        idx = AA_TO_IDX.get(ch, AA_TO_IDX["X"])  # map unknowns to 'X'
        onehot[i, idx] = 1.0

    # 2 terminal flags
    nc = np.zeros((L, 2), dtype=np.float32)
    if L > 0:
        nc[0, 0] = 1.0        # N-terminal flag
        nc[L - 1, 1] = 1.0    # C-terminal flag

    feats = np.concatenate([onehot, nc], axis=1)  # [L, 24]
    return torch.from_numpy(feats).unsqueeze(0)   # [1, L, 24]


# ---------- loaders ----------
def _load_torchscript() -> Dict[str, Any]:
    global _MODEL_TS
    if TS_PATH.exists():
        _MODEL_TS = torch.jit.load(str(TS_PATH), map_location="cpu").eval()
        return {"torchscript": str(TS_PATH)}
    return {}

def _load_checkpoint() -> Dict[str, Any]:
    global _MODEL
    if not CKPT_PATH.exists():
        return {}
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model = ProteinClassifier2(num_classes=NUM_CLASSES)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    _MODEL = model
    info = {"checkpoint": str(CKPT_PATH)}
    if "val_acc" in ckpt:
        info["val_acc"] = float(ckpt["val_acc"])
    return info

def load_model() -> Dict[str, Any]:
    """
    Load model once. Prefer TorchScript; fallback to state_dict.
    """
    info = _load_torchscript()
    if not info:
        info = _load_checkpoint()
    if not info:
        raise FileNotFoundError(
            f"No model artifacts found in {WEIGHTS_DIR}. "
            f"Expected {TS_PATH.name} or {CKPT_PATH.name} (+ meta.json)."
        )
    return info

# ---------- inference ----------
@torch.inference_mode()
def predict_secondary_structure(seq: str) -> List[str]:
    """
    seq: cleaned AA string (A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y), len <= 1000
    returns: list of labels like ["H","E","C", ...] with len(seq)
    """
    global _MODEL_TS, _MODEL
    if _MODEL_TS is None and _MODEL is None:
        load_model()

    x = _featurize(seq)  # [1,L,24]

    if _MODEL_TS is not None:
        logits = _MODEL_TS(x)     # [1,L,C]
    else:
        logits = _MODEL(x)        # [1,L,C]

    if logits.dim() == 3:
        logits = logits[0]        # [L,C]
    pred_idx = logits.argmax(dim=-1).tolist()  # [L]
    return [LABELS[i] for i in pred_idx]

def current_model_info() -> Dict[str, Any]:
    return {
        "meta": _META,
        "has_torchscript": TS_PATH.exists(),
        "has_checkpoint": CKPT_PATH.exists(),
        "weights_dir": str(WEIGHTS_DIR),
    }
