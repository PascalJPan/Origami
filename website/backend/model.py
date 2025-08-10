from typing import List

def load_model():
    # TODO: load your real model here
    return {"name": "dummy"}

def predict_secondary_structure(seq: str) -> List[str]:
    # dummy pattern (replace later)
    pat = ["H","H","H","E","E","C","H","H","C","E"]
    return [pat[i % len(pat)] for i in range(len(seq))]
