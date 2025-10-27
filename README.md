# 🏥 Health Risk Prediction System

A comprehensive web application that uses machine learning to predict diabetes and heart disease risk based on health parameters. Built with React frontend and Flask backend.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Information](#model-information)
- [Contributing](#contributing)
- [Disclaimer](#disclaimer)

## ✨ Features

- **AI-Powered Predictions**: Machine learning models for diabetes and heart disease risk assessment
- **Real-time Analysis**: Instant risk predictions with probability scores
- **User-Friendly Interface**: Modern, responsive React frontend
- **Comprehensive Health Parameters**: Multiple medical indicators for accurate assessment
- **Risk Level Classification**: Low, Medium, and High risk categories
- **Educational Information**: Detailed explanations of health factors

## 📁 Project Structure

```
health-risk-predictor/
├── data/                    # Dataset files
│   ├── pima_diabetes.csv    # Diabetes dataset
│   └── heart_disease.csv    # Heart disease dataset
├── notebooks/               # Jupyter notebooks for EDA
│   ├── eda_diabetes.ipynb   # Diabetes data exploration
│   └── eda_heart_disease.ipynb # Heart disease data exploration
├── models/                  # Trained ML models
│   ├── diabetes_lr.pkl      # Diabetes logistic regression model
│   ├── diabetes_scaler.pkl  # Diabetes feature scaler
│   ├── heart_disease_lr.pkl # Heart disease logistic regression model
│   └── heart_disease_scaler.pkl # Heart disease feature scaler
├── backend/                 # Flask API server
│   ├── app.py              # Main Flask application
│   └── requirements.txt    # Python dependencies
└── frontend/               # React application
    ├── public/             # Static files
    ├── src/                # React source code
    │   ├── components/     # React components
    │   ├── App.js          # Main App component
    │   └── index.js        # Entry point
    └── package.json        # Node.js dependencies
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Node.js 14 or higher
- npm or yarn

### Backend Setup

1. **Navigate to the backend directory:**
   ```bash
   cd health-risk-predictor/backend
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Train the models (run EDA notebooks first):**
   ```bash
   cd ../notebooks
   jupyter notebook
   ```
   - Open `eda_diabetes.ipynb` and run all cells
   - Open `eda_heart_disease.ipynb` and run all cells

6. **Start the Flask server:**
   ```bash
   cd ../backend
   python app.py
   ```
   The API will be available at `http://localhost:5000`

### Frontend Setup

1. **Navigate to the frontend directory:**
   ```bash
   cd health-risk-predictor/frontend
   ```

2. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

3. **Start the React development server:**
   ```bash
   npm start
   ```
   The application will be available at `http://localhost:3000`

## 📖 Usage

### Web Application

1. Open your browser and go to `http://localhost:3000`
2. Choose between "Diabetes Risk" or "Heart Disease Risk"
3. Fill in the required health parameters
4. Click "Predict Risk" to get your assessment
5. Review the results and risk level

### API Usage

#### Diabetes Prediction
```bash
curl -X POST http://localhost:5000/predict/diabetes \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 3,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 20,
    "Insulin": 79,
    "BMI": 32.0,
    "DiabetesPedigreeFunction": 0.3725,
    "Age": 29
  }'
```

#### Heart Disease Prediction
```bash
curl -X POST http://localhost:5000/predict/heart-disease \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 54,
    "Sex": 1,
    "ChestPainType": 1,
    "RestingBP": 130,
    "Cholesterol": 200,
    "FastingBS": 0,
    "RestingECG": 1,
    "MaxHR": 150,
    "ExerciseAngina": 0,
    "Oldpeak": 0.0,
    "ST_Slope": 1
  }'
```

## 🔌 API Documentation

### Endpoints

#### Health Check
- **GET** `/health`
- Returns API status and model loading information

#### Diabetes Prediction
- **POST** `/predict/diabetes`
- **Body**: JSON with diabetes parameters
- **Response**: Prediction results with risk level

#### Heart Disease Prediction
- **POST** `/predict/heart-disease`
- **Body**: JSON with heart disease parameters
- **Response**: Prediction results with risk level

#### Feature Information
- **GET** `/api/features/diabetes`
- **GET** `/api/features/heart-disease`
- Returns feature descriptions and valid ranges

### Response Format
```json
{
  "prediction": 0,
  "probability": 0.25,
  "risk_level": "Low",
  "message": "Diabetes risk: Low (25.0%)"
}
```

## 🤖 Model Information

### Diabetes Model
- **Algorithm**: Logistic Regression
- **Dataset**: Pima Indian Diabetes Dataset
- **Features**: 8 health parameters
- **Accuracy**: ~77% (baseline)
- **Key Factors**: Glucose levels, BMI, age, blood pressure

### Heart Disease Model
- **Algorithm**: Logistic Regression
- **Dataset**: Heart Disease Dataset
- **Features**: 11 health parameters
- **Accuracy**: ~85% (baseline)
- **Key Factors**: Age, chest pain, blood pressure, cholesterol

## 📊 Data Sources

- **Diabetes Dataset**: Pima Indian Diabetes Database
- **Heart Disease Dataset**: Heart Disease Prediction Dataset
- Both datasets are publicly available and commonly used in medical ML research

## 🔧 Development

### Running Tests
```bash
# Backend tests
cd backend
python -m pytest

# Frontend tests
cd frontend
npm test
```

### Building for Production
```bash
# Frontend build
cd frontend
npm run build

# Backend deployment
cd backend
gunicorn app:app
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## ⚠️ Important Disclaimer

**This application is for educational and demonstration purposes only.**

- The predictions provided are based on machine learning models and should not be used as a substitute for professional medical advice
- Always consult with qualified healthcare professionals for medical concerns
- The models are trained on limited datasets and may not be accurate for all populations
- This tool should be used as a screening tool only, not for diagnosis or treatment decisions

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🆘 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/your-repo/issues) page
2. Review the documentation
3. Create a new issue with detailed information

---

**Built with ❤️ for healthcare education and research**
