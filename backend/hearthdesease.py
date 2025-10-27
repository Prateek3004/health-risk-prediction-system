# ----------------- Import Necessary Libraries -----------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib

warnings.filterwarnings('ignore')

# ----------------- Load and Prepare Dataset -----------------

# Load dataset
df = pd.read_csv("C:/Users/gites/OneDrive/Desktop/python/synthetic_heart_data_50k.csv")

# Show dataset summary
print("\n--- Dataset Info ---")
print(df.info())
print("\n--- Dataset Description ---")
print(df.describe())

# Separate features and target
X = df.drop('heart_disease_present', axis=1)
y = df['heart_disease_present']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Initialize and train model
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'heart_disease_model.pkl')

# ----------------- Model Evaluation -----------------

# Predict on test set
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy on Test Set: {accuracy*100:.2f}%")

# Feature Importance
feature_importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# Target Distribution
plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title('Distribution of Heart Disease Presence')
plt.xticks([0,1], ['No Disease', 'Disease'])
plt.tight_layout()
plt.savefig('target_distribution.png')
plt.show()

print("\n✅ Training, evaluation, and graphs generation completed!")

# ----------------- Patient Prediction System -----------------

def get_patient_input():
    print("\n------ Now Enter Patient Details Below ------")
    
    age = float(input("Enter patient's Age: "))
    bp_sys = float(input("Enter Systolic Blood Pressure (e.g., 120): "))
    bp_dia = float(input("Enter Diastolic Blood Pressure (e.g., 80): "))
    cholesterol = float(input("Enter Cholesterol Level (e.g., 200): "))
    max_heart_rate = float(input("Enter Maximum Heart Rate (e.g., 150): "))
    oldpeak = float(input("Enter Oldpeak (ST depression induced by exercise, e.g., 1.2): "))
    
    gender = input("Enter Gender (Male/Female): ").strip().capitalize()
    chest_pain = input("Enter Chest Pain Type (typical_angina, atypical_angina, non_anginal_pain, asymptomatic): ").strip().lower()
    ecg_result = input("Enter ECG Result (normal, ST-T_abnormality, left_ventricular_hypertrophy): ").strip().lower()
    
    blood_sugar = int(input("Is Blood Sugar > 120 mg/dl? (1 for Yes, 0 for No): "))
    exercise_induced_angina = int(input("Exercise Induced Angina? (1 for Yes, 0 for No): "))
    smoking = int(input("Is patient a Smoker? (1 for Yes, 0 for No): "))
    diabetes = int(input("Does patient have Diabetes? (1 for Yes, 0 for No): "))
    hypertension = int(input("Does patient have Hypertension? (1 for Yes, 0 for No): "))
    family_history = int(input("Family History of Heart Disease? (1 for Yes, 0 for No): "))

    # Standardize numeric features
    scaler = StandardScaler()
    scaler.fit(df[['age', 'bp_sys', 'bp_dia', 'cholesterol', 'max_heart_rate', 'oldpeak']])
    numeric_inputs = scaler.transform(np.array([[age, bp_sys, bp_dia, cholesterol, max_heart_rate, oldpeak]]))

    # Prepare categorical features
    gender_Male = 1 if gender == 'Male' else 0

    chest_pain_typical = 1 if chest_pain == 'typical_angina' else 0
    chest_pain_atypical = 1 if chest_pain == 'atypical_angina' else 0
    chest_pain_non_anginal = 1 if chest_pain == 'non_anginal_pain' else 0

    ecg_left_ventricular = 1 if ecg_result == 'left_ventricular_hypertrophy' else 0
    ecg_normal = 1 if ecg_result == 'normal' else 0

    # Final patient input
    patient_data = np.hstack([
        numeric_inputs.flatten(),
        [gender_Male, chest_pain_atypical, chest_pain_non_anginal, chest_pain_typical,
         ecg_left_ventricular, ecg_normal,
         blood_sugar, exercise_induced_angina, smoking, diabetes, hypertension, family_history]
    ]).reshape(1, -1)
    
    return patient_data

# Load the trained model
loaded_model = joblib.load('heart_disease_model.pkl')

# Loop for multiple predictions
while True:
    patient_data = get_patient_input()
    prediction = loaded_model.predict(patient_data)
    probability = loaded_model.predict_proba(patient_data)[0]

    print("\n--- Prediction Result ---")
    print(f"Probability of NO Heart Disease: {probability[0]*100:.2f}%")
    print(f"Probability of HAVING Heart Disease: {probability[1]*100:.2f}%")
    
    if prediction[0] == 1:
        print("\n⚠️  Warning: High Risk of Heart Disease!")
    else:
        print("\n✅ Good News: Low Risk of Heart Disease!")
    
    another = input("\nWould you like to predict for another patient? (yes/no): ").strip().lower()
    if another != 'yes':
        print("\n👋 Exiting the prediction system. Stay healthy!")
        break
