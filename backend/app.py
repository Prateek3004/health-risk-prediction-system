from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import os

# ---------------------------------------------------------------------
# App Configuration
# ---------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Model Paths (Inside backend/models)
# ---------------------------------------------------------------------
MODEL_DIR = Path("models")
DATA_DIR = Path("data")

models_loaded = {
    "diabetes": False,
    "heart_disease": False,
    "general_disease": False
}

# ---------------------------------------------------------------------
# Load Models
# ---------------------------------------------------------------------
try:
    diabetes_model = joblib.load(MODEL_DIR / "diabetes_lr.pkl")
    diabetes_scaler = joblib.load(MODEL_DIR / "diabetes_scaler.pkl")
    models_loaded["diabetes"] = True
    logger.info("✅ Diabetes model loaded successfully.")
except Exception as e:
    logger.warning(f"⚠️ Failed to load Diabetes model: {e}")

try:
    heart_disease_model = joblib.load(MODEL_DIR / "heart_disease_lr.pkl")
    heart_disease_scaler = joblib.load(MODEL_DIR / "heart_disease_scaler.pkl")
    models_loaded["heart_disease"] = True
    logger.info("✅ Heart disease model loaded successfully.")
except Exception as e:
    logger.warning(f"⚠️ Failed to load Heart Disease model: {e}")

try:
    general_disease_model = joblib.load(MODEL_DIR / "general_disease_model.pkl")
    models_loaded["general_disease"] = True
    logger.info("✅ General disease model loaded successfully.")
except Exception as e:
    logger.warning(f"⚠️ Failed to load General Disease model: {e}")

# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@app.route("/")
def home():
    return jsonify({
        "message": "Welcome to the Health Risk Prediction API 🎯",
        "available_endpoints": [
            "/predict/diabetes",
            "/predict/heart",
            "/predict/general",
            "/health"
        ]
    })

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "models_loaded": models_loaded,
        "timestamp": pd.Timestamp.now().isoformat()
    })

# ---------------------------------------------------------------------
# Prediction Endpoints
# ---------------------------------------------------------------------

@app.route("/predict/diabetes", methods=["POST"])
def predict_diabetes():
    try:
        data = request.get_json()
        features = np.array(list(data.values())).reshape(1, -1)
        scaled_features = diabetes_scaler.transform(features)
        prediction = diabetes_model.predict(scaled_features)
        result = "Positive" if prediction[0] == 1 else "Negative"
        return jsonify({"prediction": result})
    except Exception as e:
        logger.error(f"Error in diabetes prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict/heart", methods=["POST"])
def predict_heart():
    try:
        data = request.get_json()
        features = np.array(list(data.values())).reshape(1, -1)
        scaled_features = heart_disease_scaler.transform(features)
        prediction = heart_disease_model.predict(scaled_features)
        result = "At Risk" if prediction[0] == 1 else "Healthy"
        return jsonify({"prediction": result})
    except Exception as e:
        logger.error(f"Error in heart prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict/general", methods=["POST"])
def predict_general():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        prediction = general_disease_model.predict(df)
        result = prediction[0]
        return jsonify({"prediction": result})
    except Exception as e:
        logger.error(f"Error in general disease prediction: {e}")
        return jsonify({"error": str(e)}), 500

# ---------------------------------------------------------------------
# Run the app
# ---------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
