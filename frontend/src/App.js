import React, { useMemo, useState } from "react";
import "./App.css";

// --- constants (keep these OUTSIDE the component) ---
const API_BASE = process.env.REACT_APP_API_BASE || ""; // empty if using CRA proxy
const AA_RE = /[ACDEFGHIKLMNPQRSTVWYX]/g;
const DUMMY_SEQ = ""; 
function generateDummyStates(n) {
  const pat = ["N"];
  return Array.from({ length: n }, (_, i) => pat[i % pat.length]);
}

export default function App() {
  // --- hooks/state (must be INSIDE the component) ---
  const [sequence, setSequence] = useState(DUMMY_SEQ);
  const [states, setStates] = useState(generateDummyStates(DUMMY_SEQ.length));
  const [indexStart, setIndexStart] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  // keep only valid AA; cap at 1000
  const cleanSequence = (raw) => {
    const up = (raw || "").toUpperCase();
    const matches = up.match(AA_RE) || [];
    return matches.join("").slice(0, 1000);
  };

  const onSeqChange = (e) => {
    const cleaned = cleanSequence(e.target.value);
    setSequence(cleaned);
    // Clear the prediction output and error when the sequence changes
    setStates(generateDummyStates(cleaned.length));  // Reset dummy states for now
    setError("");  // Clear any existing error message
  };

  const onIndexChange = (e) => {
    const v = parseInt(e.target.value, 10);
    if (Number.isNaN(v)) return setIndexStart(1);
    setIndexStart(Math.max(1, Math.min(1_000_000_000, v)));
  };

  // --- call your Python backend ---
  const runPredict = async () => {
    setIsLoading(true);
    setError("");
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sequence, index_start: indexStart }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json(); // { sequence, index_start, states }
      setSequence(data.sequence);    // reflect backend-cleaned seq
      setStates(data.states);
      setIndexStart(data.index_start);
    } catch (e) {
      setError(e.message || "Prediction failed");
    } finally {
      setIsLoading(false);
    }
  };

  const counts = useMemo(() => {
    const H = states.filter((s) => s === "H").length;
    const E = states.filter((s) => s === "E").length;
    const C = states.filter((s) => s === "C").length;
    return { H, E, C, N: sequence.length || 1 };
  }, [states, sequence.length]);

  const downloadCSV = () => {
    const header = "Index,Residue,State\n";
    const n = Math.min(sequence.length, states.length);
    const lines = Array.from({ length: n })
      .map((_, i) => `${indexStart + i},${sequence[i]},${states[i]}`);
    const blob = new Blob([header + lines.join("\n")], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "secondary_structure_prediction.csv";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const stateClass = (s) => (s === "H" ? "state-H" : s === "E" ? "state-E" : s === "C" ? "state-C" : "state-N");
  const len = Math.min(sequence.length, states.length);

  return (
    <div className="app">
      <main className="card" role="main" aria-label="Secondary structure predictor">
        <div className="header">
          <div className="logo" aria-hidden="true" />
          <div className="title">Origami — Mini Secondary Structure Predictor</div>
        </div>

        <p className="subtle">
          Paste an amino-acid sequence. Use X for unkown positions. Up to 1000 residues.
        </p>

        <div className="inputs">
          <textarea
            value={sequence}
            onChange={onSeqChange}
            spellCheck={false}
            aria-label="Amino acid sequence input"
            placeholder="Enter amino acid sequence, e.g. MVLSPADKTNVKAA..."
          />
          <div className="field">
            <label className="label" htmlFor="indexStart">Index Start</label>
            <input
              id="indexStart"
              className="input"
              type="number"
              min={1}
              step={1}
              value={indexStart}
              onChange={onIndexChange}
            />
          </div>
        </div>

        <div className="controls">
          <button className="btn" onClick={runPredict} disabled={isLoading}>
            {isLoading ? "Running…" : "Run prediction"}
          </button>
          <button className="btn" onClick={downloadCSV}>
            Download CSV
          </button>
        </div>

        {error && <div className="stat" style={{ borderColor: "#ff6b6b" }}>Error: {error}</div>}

        {/* Grid: vertical cell per residue (index on top, residue below) */}
        <div className="grid" aria-label="Sequence grid">
          {Array.from({ length: len }).map((_, i) => {
            const idx = indexStart + i;
            const tick = idx % 10 === 0 ? String(idx) : "·";
            return (
              <div key={i} className="cell" title={`Residue ${idx}: ${sequence[i]} → ${states[i]}`}>
                <div className="idx">{tick}</div>
                <div className={`res ${stateClass(states[i])}`}>{sequence[i]}</div>
              </div>
            );
          })}
        </div>

        <div className="stats" aria-label="Summary statistics">
          <div className="stat helix">Helix: <strong>{counts.H}</strong> <span>({Math.round((counts.H / counts.N) * 100)}%)</span></div>
          <div className="stat sheet">Sheet: <strong>{counts.E}</strong> <span>({Math.round((counts.E / counts.N) * 100)}%)</span></div>
          <div className="stat coil">Coil:  <strong>{counts.C}</strong> <span>({Math.round((counts.C / counts.N) * 100)}%)</span></div>
        </div>

        <div className="footer">
          <span className="small">The prediction accuracy per AA is 70% on average.</span>
        </div>
      </main>
    </div>
  );
}
