import React, { useState, useEffect } from 'react';
import axios from 'axios';

const HeartDiseasePredictor = () => {
  const [formData, setFormData] = useState(null);

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [features, setFeatures] = useState(null);
  const [schema, setSchema] = useState('lr');

  useEffect(() => {
    // Fetch feature descriptions and active schema
    const fetchFeatures = async () => {
      try {
        const response = await axios.get('/api/features/heart-disease');
        setFeatures(response.data);
        const activeSchema = response.data?.schema || 'lr';
        setSchema(activeSchema);
        if (activeSchema === 'rf') {
          setFormData({
            age: 55,
            bp_sys: 130,
            bp_dia: 80,
            cholesterol: 200,
            max_heart_rate: 150,
            oldpeak: 0.0,
            gender: 'Male',
            chest_pain: 'typical_angina',
            ecg_result: 'normal',
            blood_sugar: 0,
            exercise_induced_angina: 0,
            smoking: 0,
            diabetes: 0,
            hypertension: 0,
            family_history: 0,
          });
        } else {
          setFormData({
            Age: 54,
            Sex: 1,
            ChestPainType: 1,
            RestingBP: 130,
            Cholesterol: 200,
            FastingBS: 0,
            RestingECG: 1,
            MaxHR: 150,
            ExerciseAngina: 0,
            Oldpeak: 0.0,
            ST_Slope: 1,
          });
        }
      } catch (err) {
        console.error('Error fetching features:', err);
      }
    };
    fetchFeatures();
  }, []);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => {
      if (!prev) return prev;
      if (schema === 'rf') {
        // RF schema has a mix of strings and numeric fields
        const stringFields = new Set(['gender', 'chest_pain', 'ecg_result']);
        const int01Fields = new Set(['blood_sugar','exercise_induced_angina','smoking','diabetes','hypertension','family_history']);
        if (stringFields.has(name)) {
          return { ...prev, [name]: value };
        }
        if (int01Fields.has(name)) {
          return { ...prev, [name]: parseInt(value, 10) || 0 };
        }
        return { ...prev, [name]: parseFloat(value) || 0 };
      }
      // Legacy LR schema: all numeric
      return { ...prev, [name]: parseFloat(value) || 0 };
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post('/predict/heart-disease', formData);
      setResult(response.data);
      try {
        const entry = {
          type: 'heart',
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
          <title>Heart Disease Risk Report</title>
          ${styles}
        </head>
        <body>
          <div class="header">
            <div class="title">Heart Disease Risk Report</div>
            <div class="subtitle">Generated on ${new Date().toLocaleString()}</div>
          </div>
          <div class="section">
            <div class="row"><div class="label">Risk</div><div><span class="pill ${riskClass}">${result.risk_level}</span></div></div>
            <div class="row"><div class="label">Probability</div><div>${(result.probability * 100).toFixed(0)}%</div></div>
            <div class="row"><div class="label">Health Score</div><div>${result.health_score ?? ''}/100</div></div>
            <div class="row"><div class="label">Confidence</div><div>${result.confidence ?? ''}</div></div>
            <div class="row"><div class="label">Model Variant</div><div>${result.model_variant || 'n/a'}</div></div>
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
        <h2>❤️ Heart Disease Risk Prediction</h2>
        <p style={{ textAlign: 'center', marginBottom: '30px', color: '#666' }}>
          Enter your health parameters to get an AI-powered heart disease risk assessment
        </p>

        <form onSubmit={handleSubmit}>
          {!formData ? null : (
            <div className="form-row">
              {schema === 'rf' ? (
                <>
                  <div className="form-group">
                    <label htmlFor="age">Age (years)</label>
                    <input type="number" id="age" name="age" value={formData.age} onChange={handleInputChange} min="18" max="100" step="1" required />
                  </div>

                  <div className="form-group">
                    <label htmlFor="bp_sys">Systolic BP (mm Hg)</label>
                    <input type="number" id="bp_sys" name="bp_sys" value={formData.bp_sys} onChange={handleInputChange} min="60" max="250" step="1" required />
                  </div>

                  <div className="form-group">
                    <label htmlFor="bp_dia">Diastolic BP (mm Hg)</label>
                    <input type="number" id="bp_dia" name="bp_dia" value={formData.bp_dia} onChange={handleInputChange} min="40" max="140" step="1" required />
                  </div>

                  <div className="form-group">
                    <label htmlFor="cholesterol">Cholesterol (mg/dL)</label>
                    <input type="number" id="cholesterol" name="cholesterol" value={formData.cholesterol} onChange={handleInputChange} min="0" max="800" step="1" required />
                  </div>

                  <div className="form-group">
                    <label htmlFor="max_heart_rate">Max Heart Rate</label>
                    <input type="number" id="max_heart_rate" name="max_heart_rate" value={formData.max_heart_rate} onChange={handleInputChange} min="60" max="220" step="1" required />
                  </div>

                  <div className="form-group">
                    <label htmlFor="oldpeak">ST Depression (Oldpeak)</label>
                    <input type="number" id="oldpeak" name="oldpeak" value={formData.oldpeak} onChange={handleInputChange} min="-4" max="8" step="0.1" required />
                  </div>

                  <div className="form-group">
                    <label htmlFor="gender">Gender</label>
                    <select id="gender" name="gender" value={formData.gender} onChange={handleInputChange} required>
                      <option value="Male">Male</option>
                      <option value="Female">Female</option>
                    </select>
                  </div>

                  <div className="form-group">
                    <label htmlFor="chest_pain">Chest Pain Type</label>
                    <select id="chest_pain" name="chest_pain" value={formData.chest_pain} onChange={handleInputChange} required>
                      <option value="typical_angina">Typical Angina</option>
                      <option value="atypical_angina">Atypical Angina</option>
                      <option value="non_anginal_pain">Non-anginal Pain</option>
                      <option value="asymptomatic">Asymptomatic</option>
                    </select>
                  </div>

                  <div className="form-group">
                    <label htmlFor="ecg_result">ECG Result</label>
                    <select id="ecg_result" name="ecg_result" value={formData.ecg_result} onChange={handleInputChange} required>
                      <option value="normal">Normal</option>
                      <option value="ST-T_abnormality">ST-T Abnormality</option>
                      <option value="left_ventricular_hypertrophy">Left Ventricular Hypertrophy</option>
                    </select>
                  </div>

                  <div className="form-group">
                    <label htmlFor="blood_sugar">Blood Sugar >120 mg/dl</label>
                    <select id="blood_sugar" name="blood_sugar" value={formData.blood_sugar} onChange={handleInputChange} required>
                      <option value={0}>No</option>
                      <option value={1}>Yes</option>
                    </select>
                  </div>

                  <div className="form-group">
                    <label htmlFor="exercise_induced_angina">Exercise Induced Angina</label>
                    <select id="exercise_induced_angina" name="exercise_induced_angina" value={formData.exercise_induced_angina} onChange={handleInputChange} required>
                      <option value={0}>No</option>
                      <option value={1}>Yes</option>
                    </select>
                  </div>

                  <div className="form-group">
                    <label htmlFor="smoking">Smoking</label>
                    <select id="smoking" name="smoking" value={formData.smoking} onChange={handleInputChange} required>
                      <option value={0}>No</option>
                      <option value={1}>Yes</option>
                    </select>
                  </div>

                  <div className="form-group">
                    <label htmlFor="diabetes">Diabetes</label>
                    <select id="diabetes" name="diabetes" value={formData.diabetes} onChange={handleInputChange} required>
                      <option value={0}>No</option>
                      <option value={1}>Yes</option>
                    </select>
                  </div>

                  <div className="form-group">
                    <label htmlFor="hypertension">Hypertension</label>
                    <select id="hypertension" name="hypertension" value={formData.hypertension} onChange={handleInputChange} required>
                      <option value={0}>No</option>
                      <option value={1}>Yes</option>
                    </select>
                  </div>

                  <div className="form-group">
                    <label htmlFor="family_history">Family History</label>
                    <select id="family_history" name="family_history" value={formData.family_history} onChange={handleInputChange} required>
                      <option value={0}>No</option>
                      <option value={1}>Yes</option>
                    </select>
                  </div>
                </>
              ) : (
                <>
                  <div className="form-group">
                    <label htmlFor="Age">Age (years)</label>
                    <input type="number" id="Age" name="Age" value={formData.Age} onChange={handleInputChange} min="28" max="77" step="1" required />
                    {features && (<small style={{ color: '#666', fontSize: '0.9rem' }}>{features.features?.Age}</small>)}
                  </div>

                  <div className="form-group">
                    <label htmlFor="Sex">Sex</label>
                    <select id="Sex" name="Sex" value={formData.Sex} onChange={handleInputChange} required>
                      <option value={1}>Male</option>
                      <option value={0}>Female</option>
                    </select>
                    {features && (<small style={{ color: '#666', fontSize: '0.9rem' }}>{features.features?.Sex}</small>)}
                  </div>

                  <div className="form-group">
                    <label htmlFor="ChestPainType">Chest Pain Type</label>
                    <select id="ChestPainType" name="ChestPainType" value={formData.ChestPainType} onChange={handleInputChange} required>
                      <option value={0}>Typical Angina</option>
                      <option value={1}>Atypical Angina</option>
                      <option value={2}>Non-anginal Pain</option>
                      <option value={3}>Asymptomatic</option>
                    </select>
                    {features && (<small style={{ color: '#666', fontSize: '0.9rem' }}>{features.features?.ChestPainType}</small>)}
                  </div>

                  <div className="form-group">
                    <label htmlFor="RestingBP">Resting Blood Pressure (mm Hg)</label>
                    <input type="number" id="RestingBP" name="RestingBP" value={formData.RestingBP} onChange={handleInputChange} min="0" max="200" step="1" required />
                    {features && (<small style={{ color: '#666', fontSize: '0.9rem' }}>{features.features?.RestingBP}</small>)}
                  </div>

                  <div className="form-group">
                    <label htmlFor="Cholesterol">Cholesterol (mm/dl)</label>
                    <input type="number" id="Cholesterol" name="Cholesterol" value={formData.Cholesterol} onChange={handleInputChange} min="0" max="603" step="1" required />
                    {features && (<small style={{ color: '#666', fontSize: '0.9rem' }}>{features.features?.Cholesterol}</small>)}
                  </div>

                  <div className="form-group">
                    <label htmlFor="FastingBS">Fasting Blood Sugar > 120 mg/dl</label>
                    <select id="FastingBS" name="FastingBS" value={formData.FastingBS} onChange={handleInputChange} required>
                      <option value={0}>No</option>
                      <option value={1}>Yes</option>
                    </select>
                    {features && (<small style={{ color: '#666', fontSize: '0.9rem' }}>{features.features?.FastingBS}</small>)}
                  </div>

                  <div className="form-group">
                    <label htmlFor="RestingECG">Resting ECG Results</label>
                    <select id="RestingECG" name="RestingECG" value={formData.RestingECG} onChange={handleInputChange} required>
                      <option value={0}>Normal</option>
                      <option value={1}>ST-T Wave Abnormality</option>
                      <option value={2}>Left Ventricular Hypertrophy</option>
                    </select>
                    {features && (<small style={{ color: '#666', fontSize: '0.9rem' }}>{features.features?.RestingECG}</small>)}
                  </div>

                  <div className="form-group">
                    <label htmlFor="MaxHR">Maximum Heart Rate</label>
                    <input type="number" id="MaxHR" name="MaxHR" value={formData.MaxHR} onChange={handleInputChange} min="60" max="202" step="1" required />
                    {features && (<small style={{ color: '#666', fontSize: '0.9rem' }}>{features.features?.MaxHR}</small>)}
                  </div>

                  <div className="form-group">
                    <label htmlFor="ExerciseAngina">Exercise Induced Angina</label>
                    <select id="ExerciseAngina" name="ExerciseAngina" value={formData.ExerciseAngina} onChange={handleInputChange} required>
                      <option value={0}>No</option>
                      <option value={1}>Yes</option>
                    </select>
                    {features && (<small style={{ color: '#666', fontSize: '0.9rem' }}>{features.features?.ExerciseAngina}</small>)}
                  </div>

                  <div className="form-group">
                    <label htmlFor="Oldpeak">ST Depression (Oldpeak)</label>
                    <input type="number" id="Oldpeak" name="Oldpeak" value={formData.Oldpeak} onChange={handleInputChange} min="-2.6" max="6.2" step="0.1" required />
                    {features && (<small style={{ color: '#666', fontSize: '0.9rem' }}>{features.features?.Oldpeak}</small>)}
                  </div>

                  <div className="form-group">
                    <label htmlFor="ST_Slope">ST Slope</label>
                    <select id="ST_Slope" name="ST_Slope" value={formData.ST_Slope} onChange={handleInputChange} required>
                      <option value={0}>Upsloping</option>
                      <option value={1}>Flat</option>
                      <option value={2}>Downsloping</option>
                    </select>
                    {features && (<small style={{ color: '#666', fontSize: '0.9rem' }}>{features.features?.ST_Slope}</small>)}
                  </div>
                </>
              )}
            </div>
          )}

          <div style={{ textAlign: 'center', marginTop: '30px' }}>
            <button type="submit" className="btn btn-secondary" disabled={loading}>
              {loading ? (
                <>
                  <span className="loading"></span> Analyzing...
                </>
              ) : (
                '❤️ Predict Heart Disease Risk'
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
          <h3>Heart Disease Risk Assessment Results</h3>
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
              <li><strong>Low Risk (0-30%):</strong> Your current health parameters suggest a low risk of heart disease. Continue maintaining a healthy lifestyle.</li>
              <li><strong>Medium Risk (30-70%):</strong> Some risk factors are present. Consider lifestyle modifications and regular cardiovascular check-ups.</li>
              <li><strong>High Risk (70-100%):</strong> Multiple risk factors detected. Please consult with a cardiologist for proper evaluation.</li>
            </ul>
            <button type="button" className="btn btn-secondary" onClick={handleDownloadPDF} style={{ marginTop: '12px' }}>
              ⬇️ Download PDF Report
            </button>
          </div>
        </div>
      )}

      <div className="card">
        <h3>📋 About the Heart Disease Prediction Model</h3>
        <p>
          This heart disease risk prediction model is based on comprehensive cardiovascular datasets and uses machine learning 
          algorithms to assess your risk of developing heart disease based on various health parameters.
        </p>
        <p>
          <strong>Key factors considered:</strong>
        </p>
        <ul style={{ marginLeft: '20px', lineHeight: '1.8' }}>
          <li>Age and sex (demographic factors)</li>
          <li>Chest pain characteristics</li>
          <li>Blood pressure and cholesterol levels</li>
          <li>Fasting blood sugar levels</li>
          <li>ECG results and heart rate</li>
          <li>Exercise-induced symptoms</li>
          <li>ST segment changes</li>
        </ul>
        <div className="alert alert-info" style={{ marginTop: '20px' }}>
          <strong>Note:</strong> This is a screening tool and should not replace professional medical advice. 
          Always consult with healthcare professionals, especially cardiologists, for proper diagnosis and treatment.
        </div>
      </div>
    </div>
  );
};

export default HeartDiseasePredictor;
