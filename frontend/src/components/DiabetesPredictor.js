import React, { useState, useEffect } from 'react';
import axios from 'axios';

const DiabetesPredictor = () => {
  const [formData, setFormData] = useState({
    Pregnancies: 3,
    Glucose: 120,
    BloodPressure: 70,
    SkinThickness: 20,
    Insulin: 79,
    BMI: 32.0,
    DiabetesPedigreeFunction: 0.3725,
    Age: 29
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [features, setFeatures] = useState(null);

  useEffect(() => {
    // Fetch feature descriptions
    const fetchFeatures = async () => {
      try {
        const response = await axios.get('/api/features/diabetes');
        setFeatures(response.data);
      } catch (err) {
        console.error('Error fetching features:', err);
      }
    };
    fetchFeatures();
  }, []);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: parseFloat(value) || 0
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post('/predict/diabetes', formData);
      setResult(response.data);
      try {
        const entry = {
          type: 'diabetes',
          timestamp: Date.now(),
          probability: response.data?.probability,
          risk_level: response.data?.risk_level,
          health_score: response.data?.health_score,
        };
        const raw = localStorage.getItem('health_history');
        const arr = raw ? JSON.parse(raw) : [];
        arr.push(entry);
        localStorage.setItem('health_history', JSON.stringify(arr));
      } catch {}
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred while making the prediction');
    } finally {
      setLoading(false);
    }
  };

  const getRiskClass = (riskLevel) => {
    switch (riskLevel?.toLowerCase()) {
      case 'low': return 'risk-low';
      case 'medium': return 'risk-medium';
      case 'high': return 'risk-high';
      default: return '';
    }
  };

  const formatConfidence = (prob, label) => {
    if (typeof prob !== 'number') return label || '';
    return `${label} – ${(prob * 100).toFixed(0)}%`;
  };

  const handleDownloadPDF = () => {
    if (!result) return;
    const win = window.open('', 'PRINT', 'height=800,width=650');
    if (!win) return;
    const styles = `
      <style>
        body { font-family: Arial, sans-serif; margin: 24px; color: #222; }
        .header { text-align: center; margin-bottom: 18px; }
        .title { font-size: 20px; font-weight: 700; margin: 0; }
        .subtitle { color: #555; margin-top: 4px; }
        .section { background: #fff; border: 1px solid #eee; border-radius: 10px; padding: 16px; margin-bottom: 12px; }
        .row { display: flex; justify-content: space-between; margin: 6px 0; }
        .label { color: #555; }
        .pill { display:inline-block; padding: 4px 10px; border-radius: 999px; color: #fff; font-weight:700; font-size: 12px; }
        .pill.low { background:#4caf50; }
        .pill.medium { background:#ff9800; }
        .pill.high { background:#f44336; }
        ul { margin: 8px 0 0 18px; }
        .small { color:#777; font-size: 12px; margin-top: 16px; }
      </style>
    `;
    const riskClass = (result.risk_level || '').toLowerCase();
    const html = `
      <html>
        <head>
          <title>Diabetes Risk Report</title>
          ${styles}
        </head>
        <body>
          <div class="header">
            <div class="title">Diabetes Risk Report</div>
            <div class="subtitle">Generated on ${new Date().toLocaleString()}</div>
          </div>
          <div class="section">
            <div class="row"><div class="label">Risk</div><div><span class="pill ${riskClass}">${result.risk_level}</span></div></div>
            <div class="row"><div class="label">Probability</div><div>${(result.probability * 100).toFixed(0)}%</div></div>
            <div class="row"><div class="label">Health Score</div><div>${result.health_score ?? ''}/100</div></div>
            <div class="row"><div class="label">Confidence</div><div>${result.confidence ?? ''}</div></div>
          </div>
          ${Array.isArray(result.factors) && result.factors.length ? `
            <div class="section">
              <div style="font-weight:700;margin-bottom:6px;">Key Factors</div>
              <ul>${result.factors.map(f => `<li>${f}</li>`).join('')}</ul>
            </div>` : ''}
          ${Array.isArray(result.recommendations) && result.recommendations.length ? `
            <div class="section">
              <div style="font-weight:700;margin-bottom:6px;">Personalized Recommendations</div>
              <ul>${result.recommendations.map(r => `<li>${r}</li>`).join('')}</ul>
            </div>` : ''}
          <div class="small">This screening report is informational and not a diagnosis. Please consult a healthcare professional.</div>
          <script>window.onload = function(){ window.print(); window.close(); };</script>
        </body>
      </html>
    `;
    win.document.write(html);
    win.document.close();
  };

  return (
    <div>
      <div className="card">
        <h2>🔍 Diabetes Risk Prediction</h2>
        <p style={{ textAlign: 'center', marginBottom: '30px', color: '#666' }}>
          Enter your health parameters to get an AI-powered diabetes risk assessment
        </p>

        <form onSubmit={handleSubmit}>
          <div className="form-row">
            <div className="form-group">
              <label htmlFor="Pregnancies">Number of Pregnancies</label>
              <input
                type="number"
                id="Pregnancies"
                name="Pregnancies"
                value={formData.Pregnancies}
                onChange={handleInputChange}
                min="0"
                max="17"
                step="1"
                required
              />
              {features && (
                <small style={{ color: '#666', fontSize: '0.9rem' }}>
                  {features.features.Pregnancies}
                </small>
              )}
            </div>

            <div className="form-group">
              <label htmlFor="Glucose">Glucose Level (mg/dL)</label>
              <input
                type="number"
                id="Glucose"
                name="Glucose"
                value={formData.Glucose}
                onChange={handleInputChange}
                min="0"
                max="199"
                step="1"
                required
              />
              {features && (
                <small style={{ color: '#666', fontSize: '0.9rem' }}>
                  {features.features.Glucose}
                </small>
              )}
            </div>

            <div className="form-group">
              <label htmlFor="BloodPressure">Blood Pressure (mm Hg)</label>
              <input
                type="number"
                id="BloodPressure"
                name="BloodPressure"
                value={formData.BloodPressure}
                onChange={handleInputChange}
                min="0"
                max="122"
                step="1"
                required
              />
              {features && (
                <small style={{ color: '#666', fontSize: '0.9rem' }}>
                  {features.features.BloodPressure}
                </small>
              )}
            </div>

            <div className="form-group">
              <label htmlFor="SkinThickness">Skin Thickness (mm)</label>
              <input
                type="number"
                id="SkinThickness"
                name="SkinThickness"
                value={formData.SkinThickness}
                onChange={handleInputChange}
                min="0"
                max="99"
                step="1"
                required
              />
              {features && (
                <small style={{ color: '#666', fontSize: '0.9rem' }}>
                  {features.features.SkinThickness}
                </small>
              )}
            </div>

            <div className="form-group">
              <label htmlFor="Insulin">Insulin Level (mu U/ml)</label>
              <input
                type="number"
                id="Insulin"
                name="Insulin"
                value={formData.Insulin}
                onChange={handleInputChange}
                min="0"
                max="846"
                step="1"
                required
              />
              {features && (
                <small style={{ color: '#666', fontSize: '0.9rem' }}>
                  {features.features.Insulin}
                </small>
              )}
            </div>

            <div className="form-group">
              <label htmlFor="BMI">BMI (kg/m²)</label>
              <input
                type="number"
                id="BMI"
                name="BMI"
                value={formData.BMI}
                onChange={handleInputChange}
                min="0"
                max="67.1"
                step="0.1"
                required
              />
              {features && (
                <small style={{ color: '#666', fontSize: '0.9rem' }}>
                  {features.features.BMI}
                </small>
              )}
            </div>

            <div className="form-group">
              <label htmlFor="DiabetesPedigreeFunction">Diabetes Pedigree Function</label>
              <input
                type="number"
                id="DiabetesPedigreeFunction"
                name="DiabetesPedigreeFunction"
                value={formData.DiabetesPedigreeFunction}
                onChange={handleInputChange}
                min="0.078"
                max="2.42"
                step="0.001"
                required
              />
              {features && (
                <small style={{ color: '#666', fontSize: '0.9rem' }}>
                  {features.features.DiabetesPedigreeFunction}
                </small>
              )}
            </div>

            <div className="form-group">
              <label htmlFor="Age">Age (years)</label>
              <input
                type="number"
                id="Age"
                name="Age"
                value={formData.Age}
                onChange={handleInputChange}
                min="21"
                max="81"
                step="1"
                required
              />
              {features && (
                <small style={{ color: '#666', fontSize: '0.9rem' }}>
                  {features.features.Age}
                </small>
              )}
            </div>
          </div>

          <div style={{ textAlign: 'center', marginTop: '30px' }}>
            <button type="submit" className="btn" disabled={loading}>
              {loading ? (
                <>
                  <span className="loading"></span> Analyzing...
                </>
              ) : (
                '🔍 Predict Diabetes Risk'
              )}
            </button>
          </div>
        </form>
      </div>

      {error && (
        <div className="alert alert-error">
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <div className="result-card">
          <h3>Diabetes Risk Assessment Results</h3>
          <div className={`risk-level ${getRiskClass(result.risk_level)}`}>
            {result.risk_level} Risk
          </div>
          <div className="probability">
            {formatConfidence(result.probability, `${result.risk_level} Risk`)}
          </div>
          {typeof result.health_score === 'number' && (
            <div style={{ marginTop: '8px' }}>
              <strong>Health Score:</strong> {result.health_score} / 100
            </div>
          )}
          {Array.isArray(result.factors) && result.factors.length > 0 && (
            <div style={{ marginTop: '10px' }}>
              <strong>Key factors:</strong> {result.factors.join(', ')}
            </div>
          )}
          <p style={{ marginTop: '20px', fontSize: '1.1rem' }}>
            {result.message}
          </p>
          {Array.isArray(result.recommendations) && result.recommendations.length > 0 && (
            <div style={{ marginTop: '20px' }}>
              <h4 style={{ color: 'white', marginBottom: '10px' }}>Personalized Recommendations</h4>
              <ul style={{ marginLeft: '20px', lineHeight: '1.8' }}>
                {result.recommendations.map((rec, idx) => (
                  <li key={idx}>{rec}</li>
                ))}
              </ul>
            </div>
          )}
          
          <div style={{ marginTop: '30px', textAlign: 'left' }}>
            <h4 style={{ color: 'white', marginBottom: '15px' }}>What this means:</h4>
            <ul style={{ textAlign: 'left', lineHeight: '1.8' }}>
              <li><strong>Low Risk (0-30%):</strong> Your current health parameters suggest a low risk of diabetes. Continue maintaining a healthy lifestyle.</li>
              <li><strong>Medium Risk (30-70%):</strong> Some risk factors are present. Consider lifestyle modifications and regular health check-ups.</li>
              <li><strong>High Risk (70-100%):</strong> Multiple risk factors detected. Please consult with a healthcare professional for proper evaluation.</li>
            </ul>
            <button type="button" className="btn" onClick={handleDownloadPDF} style={{ marginTop: '12px' }}>
              ⬇️ Download PDF Report
            </button>
          </div>
        </div>
      )}

      <div className="card">
        <h3>📋 About the Diabetes Prediction Model</h3>
        <p>
          This diabetes risk prediction model is based on the Pima Indian Diabetes dataset and uses machine learning 
          algorithms to assess your risk of developing diabetes based on various health parameters.
        </p>
        <p>
          <strong>Key factors considered:</strong>
        </p>
        <ul style={{ marginLeft: '20px', lineHeight: '1.8' }}>
          <li>Glucose levels (primary indicator)</li>
          <li>Age and pregnancy history</li>
          <li>Blood pressure and BMI</li>
          <li>Insulin levels and skin thickness</li>
          <li>Family history (diabetes pedigree function)</li>
        </ul>
        <div className="alert alert-info" style={{ marginTop: '20px' }}>
          <strong>Note:</strong> This is a screening tool and should not replace professional medical advice. 
          Always consult with healthcare professionals for proper diagnosis and treatment.
        </div>
      </div>
    </div>
  );
};

export default DiabetesPredictor;
