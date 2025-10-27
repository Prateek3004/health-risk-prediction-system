#!/usr/bin/env python3
"""
Model Training Script for Health Risk Prediction System

This script trains logistic regression models for diabetes and heart disease prediction
and saves them to the models/ directory for use in the Flask API.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
import os
from pathlib import Path
import re

# Resolve project paths relative to this script so it works from any CWD
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

def train_diabetes_model():
    """Train and save the diabetes prediction model"""
    print("🔄 Training Diabetes Model...")
    
    # Load diabetes dataset
    df = pd.read_csv(DATA_DIR / "pima_diabetes.csv")
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Handle missing values (replace zeros with median for certain features)
    features_to_impute = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for feature in features_to_impute:
        if df[feature].median() != 0:
            df[feature] = df[feature].replace(0, df[feature].median())
    
    # Prepare features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model (more robust: class-balanced LR)
    model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"✅ Diabetes Model Training Complete!")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   AUC Score: {auc:.4f}")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    
    # Save model and scaler
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "diabetes_lr.pkl")
    joblib.dump(scaler, MODELS_DIR / "diabetes_scaler.pkl")
    print("💾 Diabetes model and scaler saved to models/")
    
    return model, scaler, accuracy, auc

def train_heart_disease_model():
    """Train and save the heart disease prediction model.
    Prefers updated synthetic dataset with richer features; falls back to classic dataset.
    """
    print("\n🔄 Training Heart Disease Model...")

    synthetic_path_candidates = [
        DATA_DIR / "synthetic_heart_data_50k.csv",
        BASE_DIR.parent / "data" / "synthetic_heart_data_50k.csv",
        Path("C:/Users/gites/OneDrive/Desktop/python/synthetic_heart_data_50k.csv"),
    ]

    df = None
    used_dataset = None
    for p in synthetic_path_candidates:
        if p.exists():
            df = pd.read_csv(p)
            used_dataset = str(p)
            break

    if df is not None and 'heart_disease_present' in df.columns:
        target_col = 'heart_disease_present'
    else:
        # Fallback to bundled dataset
        df = pd.read_csv(DATA_DIR / "heart_disease.csv")
        used_dataset = str(DATA_DIR / "heart_disease.csv")
        target_col = 'HeartDisease'

    print(f"Dataset loaded from: {used_dataset}")
    print(f"Samples: {df.shape[0]}, Features: {df.shape[1]}")

    # Prepare features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Stratified split when possible
    stratify_arg = y if y.nunique() == 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_arg
    )

    # Choose model: if synthetic dataset is used, train RandomForest directly on provided features
    if target_col == 'heart_disease_present':
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        print(f"✅ Heart Disease RF Training Complete!")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   AUC Score: {auc:.4f}")
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Test samples: {X_test.shape[0]}")

        # Save RF model (no scaler needed)
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(model, MODELS_DIR / "heart_disease_model.pkl")
        print("💾 Heart disease RF model saved to models/heart_disease_model.pkl")

        return model, None, accuracy, auc
    else:
        # Legacy LR path with scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        print(f"✅ Heart Disease LR Training Complete!")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   AUC Score: {auc:.4f}")
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Test samples: {X_test.shape[0]}")

        # Save model and scaler
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(model, MODELS_DIR / "heart_disease_lr.pkl")
        joblib.dump(scaler, MODELS_DIR / "heart_disease_scaler.pkl")
        print("💾 Heart disease LR model and scaler saved to models/")

        return model, scaler, accuracy, auc

def _normalize_symptom(name: str):
    if not isinstance(name, str):
        return ''
    s = name.strip().lower()
    s = s.replace('-', '_').replace(' ', '_')
    # collapse multiple underscores and remove stray commas
    s = '_'.join([t for t in s.replace(',', ' ').split('_') if t])
    return s

def train_general_disease_model():
    """Train and save a general disease classifier from symptom_Description.csv.
    Produces a multi-class Logistic Regression on binary symptom features.
    Saves a single artifact dict with: model, symptoms (feature order), label_encoder.
    """
    print("\n🔄 Training General Disease Model (Symptoms → Disease)...")
    # Try multiple candidates; pick the one that actually has Symptom columns
    candidates = [
        DATA_DIR / "generaldiseases.csv",
        DATA_DIR / "symptom_Description.csv",
    ]
    src = None
    df = None
    for cand in candidates:
        if cand.exists():
            tmp = pd.read_csv(cand)
            # Normalize temp columns and check for symptom pattern
            def _norm_col(c):
                s = str(c)
                s = re.sub(r"\s+", "", s.strip())
                s = s.replace('-', '_')
                return s
            tmp.columns = [_norm_col(c) for c in tmp.columns]
            scols = [c for c in tmp.columns if re.match(r"(?i)^symptom_?\d+$", c)]
            if scols:
                src = cand
                df = tmp
                break
    if df is None:
        raise ValueError("No symptom-mapped dataset found. Ensure a CSV with columns Symptom_1.. and a Disease column exists (e.g., generaldiseases.csv).")

    print(f"Using general disease dataset: {src}")
    # Normalize column names to be robust to stray spaces/case
    # df already normalized and verified to have symptom columns
    symptom_cols = [c for c in df.columns if re.match(r"(?i)^symptom_?\d+$", c)]

    # Normalize all symptom strings
    for c in symptom_cols:
        df[c] = df[c].apply(_normalize_symptom)

    # Build vocabulary of symptoms
    symptom_set = set()
    for c in symptom_cols:
        symptom_set.update([s for s in df[c].unique() if isinstance(s, str) and s])
    symptoms = sorted(symptom_set)

    # Vectorize rows to binary matrix
    def row_to_vector(row):
        present = set([_normalize_symptom(str(row[c])) for c in symptom_cols if isinstance(row[c], str) and row[c]])
        return [1 if s in present else 0 for s in symptoms]

    X = np.array([row_to_vector(row) for _, row in df.iterrows()], dtype=np.float32)
    y_text = df['Disease'].astype(str).values

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Multiclass Logistic Regression
    model = LogisticRegression(
        max_iter=1000,
        multi_class='auto',
        class_weight='balanced',
        n_jobs=None,
        solver='lbfgs'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ General Disease Model Training Complete!")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   Classes: {len(le.classes_)} diseases, Symptoms: {len(symptoms)}")

    # Save artifact
    os.makedirs(MODELS_DIR, exist_ok=True)
    artifact = {
        'model': model,
        'symptoms': symptoms,
        'label_encoder': le,
    }
    joblib.dump(artifact, MODELS_DIR / "general_disease_model.pkl")
    print("💾 General disease model saved to models/general_disease_model.pkl")

    return model, symptoms, le, acc

def main():
    """Main function to train both models"""
    print("🏥 Health Risk Prediction System - Model Training")
    print("=" * 50)
    
    try:
        # Train diabetes model
        diabetes_model, diabetes_scaler, diabetes_acc, diabetes_auc = train_diabetes_model()
        
        # Train heart disease model
        heart_model, heart_scaler, heart_acc, heart_auc = train_heart_disease_model()

        # Train general disease (symptom-based) model
        gen_model, gen_symptoms, gen_le, gen_acc = train_general_disease_model()
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 Training Summary")
        print("=" * 50)
        print(f"Diabetes Model:")
        print(f"  - Accuracy: {diabetes_acc:.4f}")
        print(f"  - AUC Score: {diabetes_auc:.4f}")
        print(f"  - Model saved: {MODELS_DIR / 'diabetes_lr.pkl'}")
        print(f"  - Scaler saved: {MODELS_DIR / 'diabetes_scaler.pkl'}")
        
        print(f"\nHeart Disease Model:")
        print(f"  - Accuracy: {heart_acc:.4f}")
        print(f"  - AUC Score: {heart_auc:.4f}")
        heart_variant = 'rf' if heart_scaler is None else 'lr'
        if heart_variant == 'rf':
            print(f"  - Model saved: {MODELS_DIR / 'heart_disease_model.pkl'}")
        else:
            print(f"  - Model saved: {MODELS_DIR / 'heart_disease_lr.pkl'}")
            print(f"  - Scaler saved: {MODELS_DIR / 'heart_disease_scaler.pkl'}")
        
        print(f"\nGeneral Disease Model:")
        print(f"  - Accuracy: {gen_acc:.4f}")
        print(f"  - Model saved: {MODELS_DIR / 'general_disease_model.pkl'}")
        print(f"  - Features (symptoms): {len(gen_symptoms)}")
        
        print("\n✅ All models trained and saved successfully!")
        print("🚀 You can now start the Flask API server.")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure the data files exist in the data/ directory.")
    except Exception as e:
        print(f"❌ Error during training: {e}")

if __name__ == "__main__":
    main()
