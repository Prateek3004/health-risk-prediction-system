import React, { useEffect, useMemo, useState } from 'react';

const Dashboard = () => {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    try {
      const raw = localStorage.getItem('health_history');
      if (raw) setHistory(JSON.parse(raw));
    } catch (e) {
      // ignore
    }
  }, []);

  const grouped = useMemo(() => {
    const g = { diabetes: [], heart: [], general: [] };
    history.forEach(h => {
      if (h?.type === 'diabetes') g.diabetes.push(h);
      else if (h?.type === 'heart') g.heart.push(h);
      else if (h?.type === 'general') g.general.push(h);
    });
    return g;
  }, [history]);

  const RiskBadge = ({ level }) => (
    <span className={`badge ${level ? level.toLowerCase() : ''}`}>{level}</span>
  );

  return (
    <div>
      <div className="card">
        <h2>📊 Health Dashboard</h2>
        <p style={{ color: '#666' }}>Your recent checks and simple trends.</p>
      </div>

      <div className="card">
        <h3>Diabetes Trend</h3>
        {grouped.diabetes.length === 0 ? (
          <p style={{ color: '#999' }}>No diabetes checks yet.</p>
        ) : (
          <div className="trend">
            <div className="trend-bars">
              {grouped.diabetes.slice(-12).map((d, idx) => (
                <div key={idx} className={`bar ${d.risk_level.toLowerCase()}`} style={{ height: `${d.health_score || (100 - (d.probability||0)*100)}px` }} title={`${new Date(d.timestamp).toLocaleString()} – ${Math.round((d.probability||0)*100)}%`} />
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="card">
        <h3>Heart Disease Trend</h3>
        {grouped.heart.length === 0 ? (
          <p style={{ color: '#999' }}>No heart checks yet.</p>
        ) : (
          <div className="trend">
            <div className="trend-bars">
              {grouped.heart.slice(-12).map((d, idx) => (
                <div key={idx} className={`bar ${d.risk_level.toLowerCase()}`} style={{ height: `${d.health_score || (100 - (d.probability||0)*100)}px` }} title={`${new Date(d.timestamp).toLocaleString()} – ${Math.round((d.probability||0)*100)}%`} />
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="card">
        <h3>General Disease Checks</h3>
        {grouped.general.length === 0 ? (
          <p style={{ color: '#999' }}>No general disease checks yet.</p>
        ) : (
          <ul style={{ listStyle: 'none', padding: 0 }}>
            {grouped.general.slice(-12).reverse().map((d, idx) => (
              <li key={idx} className="history-item">
                <div>
                  <strong>General</strong> – {new Date(d.timestamp).toLocaleString()}
                </div>
                <div>
                  Prediction: {d.prediction}
                </div>
                {Array.isArray(d.topk) && d.topk.length > 0 && (
                  <div style={{ color: '#666' }}>
                    Top: {d.topk.slice(0,3).map(t => `${t.disease} (${Math.round((t.probability||0)*100)}%)`).join(', ')}
                  </div>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="card">
        <h3>Recent Checks</h3>
        {history.length === 0 ? (
          <p style={{ color: '#999' }}>No history yet. Make a prediction to see it here.</p>
        ) : (
          <ul style={{ listStyle: 'none', padding: 0 }}>
            {history.slice(-20).reverse().map((h, idx) => (
              <li key={idx} className="history-item">
                <div>
                  <strong>{h.type === 'diabetes' ? 'Diabetes' : h.type === 'heart' ? 'Heart' : 'General'}</strong> – {new Date(h.timestamp).toLocaleString()}
                </div>
                {h.type !== 'general' ? (
                  <div>
                    <RiskBadge level={h.risk_level} /> &nbsp; Score: {h.health_score ?? Math.round((1 - (h.probability||0)) * 100)}/100
                  </div>
                ) : (
                  <div>
                    Prediction: {h.prediction}
                  </div>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

export default Dashboard;


