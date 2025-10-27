from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ---------------- Recommendation Helpers -----------------

def _limit_recommendations(items, limit=6):
    deduped = []
    for it in items:
        if it and it not in deduped:
            deduped.append(it)
        if len(deduped) >= limit:
            break
    return deduped

def generate_diabetes_recommendations(data: dict, risk_level: str):
    recs = []
    try:
        glucose = float(data.get('Glucose', 0))
        bmi = float(data.get('BMI', 0))
        bp = float(data.get('BloodPressure', 0))
        insulin = float(data.get('Insulin', 0))
        age = float(data.get('Age', 0))
        dpf = float(data.get('DiabetesPedigreeFunction', 0))
    except Exception:
        glucose = bmi = bp = insulin = age = dpf = 0.0

    # Base by risk level
    if risk_level == 'High':
        recs += [
            'Schedule a comprehensive check-up with a healthcare provider within 1-2 weeks.',
            'Monitor fasting and post-meal blood glucose more frequently (as advised by your doctor).',
            'Adopt a structured meal plan emphasizing whole grains, lean protein, and high-fiber vegetables.',
        ]
    elif risk_level == 'Medium':
        recs += [
            'Increase daily physical activity to at least 150 minutes per week of moderate exercise.',
            'Reduce refined sugars and processed carbs; focus on balanced plate portions.',
        ]
    else:  # Low
        recs += [
            'Maintain current healthy habits and schedule annual screenings.',
        ]

    # Personalized rules
    if glucose >= 140:
        recs.append('Limit high-glycemic foods and pair carbs with protein/fiber to blunt glucose spikes.')
    if bmi >= 30:
        recs.append('Aim for gradual weight reduction (5-7%) through calorie-aware meal planning and activity.')
    if bp >= 80:
        recs.append('Lower sodium intake (<1500 mg/day) and prioritize potassium-rich foods to support blood pressure.')
    if insulin >= 200:
        recs.append('Discuss insulin resistance strategies (time-in-range goals, carb timing) with your clinician.')
    if age >= 45:
        recs.append('Ensure regular HbA1c testing every 3-6 months depending on your provider’s advice.')
    if dpf >= 0.8:
        recs.append('With elevated family risk, keep consistent lifestyle routines and routine screenings.')

    recs.append('Prioritize 7-8 hours of sleep and stress management (breathing exercises, short walks).')
    return _limit_recommendations(recs)

def generate_heart_recommendations(data: dict, risk_level: str, variant: str):
    recs = []
    # Normalize keys across schemas
    def getf(*names, default=0.0):
        for n in names:
            if n in data:
                try:
                    return float(data[n])
                except Exception:
                    pass
        return default
    def geti(*names, default=0):
        for n in names:
            if n in data:
                try:
                    return int(data[n])
                except Exception:
                    pass
        return default
    def gets(*names, default=''):
        for n in names:
            v = str(data.get(n, '')).strip().lower()
            if v:
                return v
        return default

    bp_sys = getf('bp_sys', 'RestingBP')
    bp_dia = getf('bp_dia', default=0)
    chol = getf('cholesterol', 'Cholesterol')
    maxhr = getf('max_heart_rate', 'MaxHR')
    oldpeak = getf('oldpeak', 'Oldpeak')
    fasting_bs = geti('blood_sugar', 'FastingBS')
    smoking = geti('smoking')
    diabetes = geti('diabetes')
    hypertension = geti('hypertension')
    exercise_angina = geti('exercise_induced_angina', 'ExerciseAngina')
    gender = gets('gender')

    # Base by risk level
    if risk_level == 'High':
        recs += [
            'Consult a cardiologist promptly for a tailored management plan.',
            'Avoid strenuous activity until medically cleared; choose gentle walks instead.',
            'Review medications and risk factors with your doctor (cholesterol, blood pressure, diabetes).',
        ]
    elif risk_level == 'Medium':
        recs += [
            'Target 150–300 minutes/week of moderate aerobic activity plus 2 strength sessions.',
            'Adopt a heart-healthy pattern (e.g., DASH/Mediterranean) rich in vegetables and healthy fats.',
        ]
    else:
        recs += [
            'Maintain routine physical activity and balanced diet; keep annual heart check-ups.',
        ]

    # Personalized rules
    if bp_sys >= 130 or bp_dia >= 80 or hypertension == 1:
        recs.append('Reduce sodium (<1500 mg/day), increase potassium, and monitor home blood pressure.')
    if chol >= 200:
        recs.append('Increase soluble fiber (oats, beans) and omega-3s; discuss lipid panel follow-up.')
    if smoking == 1:
        recs.append('Enroll in a smoking cessation program; consider nicotine replacement or counseling.')
    if diabetes == 1 or fasting_bs == 1:
        recs.append('Optimize glucose control; coordinate care between cardiology and primary care/endocrinology.')
    if exercise_angina == 1 or oldpeak > 0:
        recs.append('Start with low-intensity, short-duration walks and slowly progress under guidance.')
    if maxhr < 100:
        recs.append('Ask your doctor about safe target heart rate zones before exercising.')
    if gender == 'male':
        recs.append('Men: consider earlier screening for lipids and blood pressure if additional risks exist.')

    recs.append('Prioritize sleep, stress reduction, and regular follow-up to track improvements.')
    return _limit_recommendations(recs)

def compute_health_score_and_confidence(probability: float):
    # probability is risk of positive class
    probability = float(max(0.0, min(1.0, probability)))
    health_score = int(round((1.0 - probability) * 100))
    if probability < 0.33:
        confidence = 'High'
    elif probability < 0.66:
        confidence = 'Medium'
    else:
        confidence = 'High'
    return health_score, confidence

def explain_diabetes_factors(data: dict):
    factors = []
    try:
        if float(data.get('BMI', 0)) >= 30:
            factors.append('Elevated BMI')
        if float(data.get('Glucose', 0)) >= 140:
            factors.append('High glucose level')
        if float(data.get('BloodPressure', 0)) >= 80:
            factors.append('Higher blood pressure')
        if float(data.get('Insulin', 0)) >= 200:
            factors.append('Elevated insulin')
        if float(data.get('DiabetesPedigreeFunction', 0)) >= 0.8:
            factors.append('Family risk (pedigree)')
        if float(data.get('Age', 0)) >= 45:
            factors.append('Older age')
    except Exception:
        pass
    if not factors:
        factors.append('Overall parameters within typical ranges')
    return factors[:4]

# ---------------- General Disease (Symptoms) Helpers -----------------

def _normalize_symptom(name: str):
    try:
        s = str(name).strip().lower()
    except Exception:
        return ''
    s = s.replace('-', '_').replace(' ', '_')
    s = '_'.join([t for t in s.replace(',', ' ').split('_') if t])
    return s

# Load general disease model artifact
general_artifact = None
general_symptoms = []
general_label_encoder = None
general_descriptions = {}

try:
    artifact_path_candidates = [
        '../models/general_disease_model.pkl',
        'models/general_disease_model.pkl',
    ]
    for p in artifact_path_candidates:
        if os.path.exists(p):
            general_artifact = joblib.load(p)
            break
    if general_artifact is not None:
        general_model = general_artifact.get('model')
        general_symptoms = general_artifact.get('symptoms', [])
        general_label_encoder = general_artifact.get('label_encoder')
        logger.info(f"General disease model loaded with {len(general_symptoms)} symptoms and {len(getattr(general_label_encoder, 'classes_', []))} classes")
    else:
        general_model = None
        logger.warning("General disease model artifact not found.")
except Exception as e:
    general_model = None
    logger.error(f"Failed to load general disease model: {e}")

# Load disease descriptions if available
try:
    desc_path = None
    for candidate in [
        Path('../data/generaldiseases.csv'),
        Path('data/generaldiseases.csv'),
    ]:
        if candidate.exists():
            desc_path = candidate
            break
    if desc_path:
        df_desc = pd.read_csv(desc_path)
        if {'Disease', 'Description'}.issubset(df_desc.columns):
            for _, r in df_desc.iterrows():
                try:
                    general_descriptions[str(r['Disease']).strip()] = str(r['Description']).strip()
                except Exception:
                    pass
            logger.info(f"Loaded {len(general_descriptions)} disease descriptions from {desc_path}")
except Exception as e:
    logger.warning(f"Could not load disease descriptions: {e}")

def explain_heart_factors(data: dict, variant: str):
    factors = []
    def getf(*names, default=0.0):
        for n in names:
            if n in data:
                try:
                    return float(data[n])
                except Exception:
                    pass
        return default
    def geti(*names, default=0):
        for n in names:
            if n in data:
                try:
                    return int(data[n])
                except Exception:
                    pass
        return default

    bp_sys = getf('bp_sys', 'RestingBP')
    bp_dia = getf('bp_dia', default=0)
    chol = getf('cholesterol', 'Cholesterol')
    maxhr = getf('max_heart_rate', 'MaxHR')
    oldpeak = getf('oldpeak', 'Oldpeak')
    fasting_bs = geti('blood_sugar', 'FastingBS')
    smoking = geti('smoking')
    diabetes = geti('diabetes')
    hypertension = geti('hypertension')
    exercise_angina = geti('exercise_induced_angina', 'ExerciseAngina')

    if bp_sys >= 130 or bp_dia >= 80:
        factors.append('Elevated blood pressure')
    if chol >= 200:
        factors.append('High cholesterol')
    if oldpeak > 0:
        factors.append('ST depression (oldpeak)')
    if fasting_bs == 1 or diabetes == 1:
        factors.append('High fasting blood sugar/diabetes')
    if smoking == 1:
        factors.append('Smoking')
    if exercise_angina == 1:
        factors.append('Exercise-induced angina')
    if maxhr < 100:
        factors.append('Lower maximum heart rate')
    if not factors:
        factors.append('No strong adverse factors detected')
    return factors[:4]

# Load models and scalers
# Diabetes
diabetes_model = None
diabetes_scaler = None
try:
    diabetes_model = joblib.load('../models/diabetes_lr.pkl')
    diabetes_scaler = joblib.load('../models/diabetes_scaler.pkl')
    logger.info("Diabetes model loaded successfully (LR)")
except FileNotFoundError:
    logger.warning("Diabetes model files not found in ../models. Skipping diabetes load.")

# Heart disease: prefer the new RandomForest model from hearthdesease.py if present
heart_model = None
heart_scaler = None  # Not needed for RF path, kept for backward compat
heart_model_paths = [
    '../models/heart_disease_model.pkl',
    'heart_disease_model.pkl',
]
heart_lr_paths = [
    '../models/heart_disease_lr.pkl'
]

loaded_heart_variant = None
for p in heart_model_paths:
    if os.path.exists(p):
        try:
            heart_model = joblib.load(p)
            loaded_heart_variant = 'rf'
            logger.info(f"Heart disease model loaded (RF) from {p}")
            break
        except Exception as e:
            logger.error(f"Failed loading heart RF model from {p}: {e}")

if heart_model is None:
    for p in heart_lr_paths:
        if os.path.exists(p):
            try:
                heart_model = joblib.load(p)
                heart_scaler = joblib.load('../models/heart_disease_scaler.pkl')
                loaded_heart_variant = 'lr'
                logger.info(f"Heart disease model loaded (LR) from {p}")
                break
            except Exception as e:
                logger.error(f"Failed loading heart LR model from {p}: {e}")

if heart_model is None:
    logger.warning("No heart disease model available. Provide either heart_disease_model.pkl or heart_disease_lr.pkl.")

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        "message": "Health Risk Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "diabetes": "/predict/diabetes",
            "heart_disease": "/predict/heart-disease",
            "general_disease": "/predict/general-disease",
            "health": "/health"
        }
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "diabetes": diabetes_model is not None,
            "heart_disease": heart_model is not None,
            "general_disease": general_artifact is not None
        }
    })

@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    """Predict diabetes risk"""
    if diabetes_model is None or diabetes_scaler is None:
        return jsonify({
            "error": "Diabetes model not loaded. Please run the EDA notebook first."
        }), 500
    
    try:
        data = request.get_json(force=True) or {}
        
        # Validate required fields
        required_fields = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                          'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({"error": "Missing required fields", "fields": missing}), 400

        def to_float(x, name):
            try:
                return float(x)
            except Exception:
                raise ValueError(f"Field {name} must be numeric")
        
        # Create feature array
        features = np.array([[
            to_float(data['Pregnancies'], 'Pregnancies'),
            to_float(data['Glucose'], 'Glucose'),
            to_float(data['BloodPressure'], 'BloodPressure'),
            to_float(data['SkinThickness'], 'SkinThickness'),
            to_float(data['Insulin'], 'Insulin'),
            to_float(data['BMI'], 'BMI'),
            to_float(data['DiabetesPedigreeFunction'], 'DiabetesPedigreeFunction'),
            to_float(data['Age'], 'Age')
        ]])
        
        # Scale features
        features_scaled = diabetes_scaler.transform(features)
        
        # Make prediction
        prediction = diabetes_model.predict(features_scaled)[0]
        probability = diabetes_model.predict_proba(features_scaled)[0][1]
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        health_score, confidence = compute_health_score_and_confidence(probability)
        factors = explain_diabetes_factors(data)

        return jsonify({
            "prediction": int(prediction),
            "probability": float(probability),
            "risk_level": risk_level,
            "message": f"Diabetes risk: {risk_level} ({probability:.1%})",
            "health_score": health_score,
            "confidence": confidence,
            "factors": factors,
            "recommendations": generate_diabetes_recommendations(data, risk_level)
        })
        
    except Exception as e:
        logger.error(f"Error in diabetes prediction: {str(e)}")
        return jsonify({"error": str(e) or "Invalid input data"}), 400

# ---------------- General Disease (Symptoms) Endpoints -----------------

def generate_general_recommendations(selected_symptoms: list, disease_label: str):
    recs = []
    sset = set([_normalize_symptom(s) for s in selected_symptoms or []])
    d = (disease_label or '').strip().lower()

    # Disease-specific nudges (very generic and non-diagnostic)
    if 'malaria' in d or 'dengue' in d:
        recs += [
            'Hydrate adequately and monitor temperature regularly.',
            'Seek medical evaluation for fever with body aches or rash.',
        ]
    if 'pneumonia' in d or 'bronchial asthma' in d or 'asthma' in d:
        recs += [
            'Avoid respiratory irritants and rest; seek clinical review if breathing is difficult.',
        ]
    if 'jaundice' in d or 'hepatitis' in d:
        recs += [
            'Avoid alcohol and hepatotoxic drugs; prioritize medical consultation and liver tests.',
        ]
    if 'gerd' in d or 'ulcer' in d or 'gastroenteritis' in d:
        recs += [
            'Prefer small, non-spicy meals; maintain hydration; avoid late heavy meals.',
        ]
    if 'allergy' in d or 'drug reaction' in d or 'urticaria' in d:
        recs += [
            'Identify and avoid suspected triggers; seek care for severe reactions.',
        ]

    # Symptom-based commonsense advice
    if any(s in sset for s in ['fever', 'high_fever', 'chills', 'shivering']):
        recs.append('Rest, hydrate, and monitor fever; consult a clinician if persistent >48 hours.')
    if any(s in sset for s in ['chest_pain']):
        recs.append('Chest pain requires prompt clinical assessment; seek urgent care if severe.')
    if any(s in sset for s in ['vomiting', 'diarrhoea', 'diarrhea', 'nausea']):
        recs.append('Use oral rehydration solutions; seek care if signs of dehydration appear.')
    if any(s in sset for s in ['itching', 'skin_rash']):
        recs.append('Avoid scratching; consider antihistamine after clinician advice; keep skin moisturized.')
    if any(s in sset for s in ['yellowish_skin', 'yellowing_of_eyes']):
        recs.append('Consider medical evaluation for possible liver involvement.')

    # Always include general disclaimer advice
    recs.append('This is not a diagnosis. Please consult a qualified healthcare professional.')
    # Deduplicate and cap
    return _limit_recommendations([r for r in recs if r])

@app.route('/api/features/general-disease', methods=['GET'])
def get_general_disease_features():
    """Return available symptoms and diseases list."""
    return jsonify({
        "symptoms": general_symptoms,
        "num_symptoms": len(general_symptoms),
        "diseases": list(getattr(general_label_encoder, 'classes_', [])),
        "num_diseases": int(len(getattr(general_label_encoder, 'classes_', [])))
    })

@app.route('/predict/general-disease', methods=['POST'])
def predict_general_disease():
    if general_artifact is None:
        return jsonify({"error": "General disease model not loaded. Please run training first."}), 500
    try:
        payload = request.get_json(force=True) or {}
        # Accept either a list under 'symptoms' or a comma-separated string under 'input'
        raw_list = payload.get('symptoms')
        if raw_list is None and 'input' in payload:
            raw_list = [s.strip() for s in str(payload.get('input', '')).split(',') if s.strip()]
        if not isinstance(raw_list, list):
            return jsonify({"error": "Provide symptoms as a list under key 'symptoms' or as comma-separated 'input'"}), 400

        norm = set([_normalize_symptom(s) for s in raw_list if s])
        vec = np.array([[1 if s in norm else 0 for s in general_symptoms]], dtype=float)

        proba = None
        if hasattr(general_artifact['model'], 'predict_proba'):
            proba = general_artifact['model'].predict_proba(vec)[0]
        pred_idx = int(general_artifact['model'].predict(vec)[0])
        pred_label = str(general_label_encoder.inverse_transform([pred_idx])[0])

        topk = []
        if proba is not None:
            order = np.argsort(proba)[::-1]
            for i in order[:5]:
                lbl = str(general_label_encoder.inverse_transform([i])[0])
                topk.append({
                    "disease": lbl,
                    "probability": float(proba[i])
                })

        description = general_descriptions.get(pred_label)
        recommendations = generate_general_recommendations(sorted(list(norm)), pred_label)

        return jsonify({
            "prediction": pred_label,
            "description": description,
            "topk": topk,
            "selected_symptoms": sorted(list(norm)),
            "vector_length": len(general_symptoms),
            "recommendations": recommendations
        })
    except Exception as e:
        logger.error(f"Error in general disease prediction: {e}")
        return jsonify({"error": str(e) or "Invalid input data"}), 400

@app.route('/predict/heart-disease', methods=['POST'])
def predict_heart_disease():
    """Predict heart disease risk"""
    if heart_model is None:
        return jsonify({
            "error": "Heart disease model not loaded. Please run training or provide a model file."
        }), 500
    
    try:
        data = request.get_json(force=True) or {}

        def to_float(x, name):
            try:
                return float(x)
            except Exception:
                raise ValueError(f"Field {name} must be numeric")

        if 'loaded_heart_variant' in globals() and loaded_heart_variant == 'rf':
            required = [
                'age','bp_sys','bp_dia','cholesterol','max_heart_rate','oldpeak',
                'gender','chest_pain','ecg_result','blood_sugar','exercise_induced_angina',
                'smoking','diabetes','hypertension','family_history'
            ]
            missing = [f for f in required if f not in data]
            if missing:
                return jsonify({"error": "Missing required fields", "fields": missing, "schema": "rf"}), 400

            numeric = [
                to_float(data['age'], 'age'),
                to_float(data['bp_sys'], 'bp_sys'),
                to_float(data['bp_dia'], 'bp_dia'),
                to_float(data['cholesterol'], 'cholesterol'),
                to_float(data['max_heart_rate'], 'max_heart_rate'),
                to_float(data['oldpeak'], 'oldpeak')
            ]

            gender = str(data['gender']).strip().lower()
            chest = str(data['chest_pain']).strip().lower()
            ecg = str(data['ecg_result']).strip().lower()

            gender_Male = 1 if gender == 'male' else 0
            chest_pain_typical = 1 if chest == 'typical_angina' else 0
            chest_pain_atypical = 1 if chest == 'atypical_angina' else 0
            chest_pain_non_anginal = 1 if chest == 'non_anginal_pain' else 0
            ecg_left_ventricular = 1 if ecg == 'left_ventricular_hypertrophy' else 0
            ecg_normal = 1 if ecg == 'normal' else 0

            def to_int01(x, name):
                v = int(x)
                if v not in (0, 1):
                    raise ValueError(f"Field {name} must be 0 or 1")
                return v

            tail = [
                to_int01(data['blood_sugar'], 'blood_sugar'),
                to_int01(data['exercise_induced_angina'], 'exercise_induced_angina'),
                to_int01(data['smoking'], 'smoking'),
                to_int01(data['diabetes'], 'diabetes'),
                to_int01(data['hypertension'], 'hypertension'),
                to_int01(data['family_history'], 'family_history'),
            ]

            features = np.hstack([
                np.array(numeric, dtype=float),
                np.array([
                    gender_Male,
                    chest_pain_atypical,
                    chest_pain_non_anginal,
                    chest_pain_typical,
                    ecg_left_ventricular,
                    ecg_normal,
                ], dtype=float),
                np.array(tail, dtype=float)
            ]).reshape(1, -1)

            features_for_inference = features
        else:
            required_fields = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                              'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
                              'Oldpeak', 'ST_Slope']
            missing = [f for f in required_fields if f not in data]
            if missing:
                return jsonify({"error": "Missing required fields", "fields": missing, "schema": "lr"}), 400

            if heart_scaler is None:
                return jsonify({"error": "Heart scaler not loaded for legacy LR model."}), 500

            features = np.array([[
                to_float(data['Age'], 'Age'),
                to_float(data['Sex'], 'Sex'),
                to_float(data['ChestPainType'], 'ChestPainType'),
                to_float(data['RestingBP'], 'RestingBP'),
                to_float(data['Cholesterol'], 'Cholesterol'),
                to_float(data['FastingBS'], 'FastingBS'),
                to_float(data['RestingECG'], 'RestingECG'),
                to_float(data['MaxHR'], 'MaxHR'),
                to_float(data['ExerciseAngina'], 'ExerciseAngina'),
                to_float(data['Oldpeak'], 'Oldpeak'),
                to_float(data['ST_Slope'], 'ST_Slope')
            ]])

            features_for_inference = heart_scaler.transform(features)

        prediction = int(heart_model.predict(features_for_inference)[0])
        proba = heart_model.predict_proba(features_for_inference)[0]
        probability = float(proba[1] if proba.shape[0] > 1 else proba[0])
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return jsonify({
            "prediction": int(prediction),
            "probability": float(probability),
            "risk_level": risk_level,
            "model_variant": (loaded_heart_variant if 'loaded_heart_variant' in globals() else None),
            "message": f"Heart disease risk: {risk_level} ({probability:.1%})",
            "health_score": compute_health_score_and_confidence(probability)[0],
            "confidence": compute_health_score_and_confidence(probability)[1],
            "factors": explain_heart_factors(data, (loaded_heart_variant if 'loaded_heart_variant' in globals() else 'lr')),
            "recommendations": generate_heart_recommendations(data, risk_level, (loaded_heart_variant if 'loaded_heart_variant' in globals() else 'lr'))
        })
        
    except Exception as e:
        logger.error(f"Error in heart disease prediction: {str(e)}")
        return jsonify({"error": str(e) or "Invalid input data"}), 400

@app.route('/api/features/diabetes', methods=['GET'])
def get_diabetes_features():
    """Get diabetes feature descriptions"""
    return jsonify({
        "features": {
            "Pregnancies": "Number of times pregnant",
            "Glucose": "Plasma glucose concentration (mg/dL)",
            "BloodPressure": "Diastolic blood pressure (mm Hg)",
            "SkinThickness": "Triceps skin fold thickness (mm)",
            "Insulin": "2-Hour serum insulin (mu U/ml)",
            "BMI": "Body mass index (weight in kg/(height in m)^2)",
            "DiabetesPedigreeFunction": "Diabetes pedigree function",
            "Age": "Age in years"
        },
        "ranges": {
            "Pregnancies": {"min": 0, "max": 17, "default": 3},
            "Glucose": {"min": 0, "max": 199, "default": 120},
            "BloodPressure": {"min": 0, "max": 122, "default": 70},
            "SkinThickness": {"min": 0, "max": 99, "default": 20},
            "Insulin": {"min": 0, "max": 846, "default": 79},
            "BMI": {"min": 0, "max": 67.1, "default": 32.0},
            "DiabetesPedigreeFunction": {"min": 0.078, "max": 2.42, "default": 0.3725},
            "Age": {"min": 21, "max": 81, "default": 29}
        }
    })

@app.route('/api/features/heart-disease', methods=['GET'])
def get_heart_disease_features():
    """Get heart disease feature descriptions"""
    if 'loaded_heart_variant' in globals() and loaded_heart_variant == 'rf':
        return jsonify({
            "schema": "rf",
            "features": {
                "age": "Age in years",
                "bp_sys": "Systolic blood pressure (mm Hg)",
                "bp_dia": "Diastolic blood pressure (mm Hg)",
                "cholesterol": "Cholesterol level (mg/dL)",
                "max_heart_rate": "Maximum heart rate achieved",
                "oldpeak": "ST depression induced by exercise",
                "gender": "Male/Female",
                "chest_pain": "typical_angina | atypical_angina | non_anginal_pain | asymptomatic",
                "ecg_result": "normal | ST-T_abnormality | left_ventricular_hypertrophy",
                "blood_sugar": ">120 mg/dl? 1=yes, 0=no",
                "exercise_induced_angina": "1=yes, 0=no",
                "smoking": "1=yes, 0=no",
                "diabetes": "1=yes, 0=no",
                "hypertension": "1=yes, 0=no",
                "family_history": "1=yes, 0=no"
            }
        })
    else:
        return jsonify({
            "schema": "lr",
            "features": {
                "Age": "Age in years",
                "Sex": "Sex (1 = male; 0 = female)",
                "ChestPainType": "Chest pain type (0-3)",
                "RestingBP": "Resting blood pressure (mm Hg)",
                "Cholesterol": "Serum cholesterol (mm/dl)",
                "FastingBS": "Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)",
                "RestingECG": "Resting electrocardiographic results (0-2)",
                "MaxHR": "Maximum heart rate achieved",
                "ExerciseAngina": "Exercise induced angina (1 = yes; 0 = no)",
                "Oldpeak": "ST depression induced by exercise relative to rest",
                "ST_Slope": "Slope of the peak exercise ST segment (0-2)"
            }
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
