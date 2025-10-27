import React, { useEffect, useMemo, useState } from 'react';

const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:5000';

function GeneralDiseasePredictor() {
  const [allSymptoms, setAllSymptoms] = useState([]);
  const [query, setQuery] = useState('');
  const [selected, setSelected] = useState([]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    async function load() {
      try {
        const res = await fetch(`${API_BASE}/api/features/general-disease`);
        const data = await res.json();
        setAllSymptoms(data.symptoms || []);
      } catch (e) {
        setError('Failed to load symptoms. Is the backend running?');
      }
    }
    load();
  }, []);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return allSymptoms.slice(0, 50);
    return allSymptoms.filter(s => s.toLowerCase().includes(q)).slice(0, 50);
  }, [allSymptoms, query]);

  const toggleSymptom = (s) => {
    if (selected.includes(s)) {
      setSelected(selected.filter(x => x !== s));
    } else {
      setSelected([...selected, s]);
    }
  };

  const submit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const res = await fetch(`${API_BASE}/predict/general-disease`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symptoms: selected })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Prediction failed');
      setResult(data);
      // Save to local history
      try {
        const entry = {
          type: 'general',
          timestamp: Date.now(),
          prediction: data.prediction,
          topk: Array.isArray(data.topk) ? data.topk.slice(0, 5) : [],
          selected_symptoms: selected,
        };
        const raw = localStorage.getItem('health_history');
        const arr = raw ? JSON.parse(raw) : [];
        arr.push(entry);
        localStorage.setItem('health_history', JSON.stringify(arr));
      } catch {}
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h2>General Disease Prediction (by Symptoms)</h2>
      <p>Select symptoms and submit to predict a likely disease. Not for medical use.</p>

      <form onSubmit={submit}>
        <div className="form-group">
          <label>Search symptoms</label>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g., skin_rash"
          />
        </div>

        <div className="symptom-list" style={{ maxHeight: 240, overflow: 'auto', border: '1px solid #ddd', padding: 8, borderRadius: 6 }}>
          {filtered.map((s) => (
            <label key={s} style={{ display: 'inline-block', margin: '4px 8px' }}>
              <input type="checkbox" checked={selected.includes(s)} onChange={() => toggleSymptom(s)} />{' '}
              {s}
            </label>
          ))}
          {filtered.length === 0 && <div>No symptoms match your search.</div>}
        </div>

        <div style={{ marginTop: 12 }}>
          <button type="submit" disabled={loading || selected.length === 0}>
            {loading ? 'Predicting…' : 'Predict Disease'}
          </button>
          <span style={{ marginLeft: 12, color: '#666' }}>Selected: {selected.length}</span>
        </div>
      </form>

      {error && <div className="error" style={{ color: 'crimson', marginTop: 12 }}>{error}</div>}

      {result && (
        <div className="result" style={{ marginTop: 16 }}>
          <h3>Prediction: {result.prediction}</h3>
          {result.description && <p style={{ whiteSpace: 'pre-wrap' }}>{result.description}</p>}
          {Array.isArray(result.recommendations) && result.recommendations.length > 0 && (
            <div style={{ marginTop: 12 }}>
              <h4>Recommendations</h4>
              <ul>
                {result.recommendations.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
            </div>
          )}
          {Array.isArray(result.topk) && result.topk.length > 0 && (
            <div>
              <h4>Top candidates</h4>
              <ul>
                {result.topk.map((k) => (
                  <li key={k.disease}>{k.disease} — {(k.probability * 100).toFixed(1)}%</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default GeneralDiseasePredictor;


