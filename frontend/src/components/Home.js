import React from 'react';
import { Link } from 'react-router-dom';

const Home = () => {
  return (
    <div>
      <div className="hero">
        <h1>🏥 Health Risk Prediction System</h1>
        <p>
          Advanced machine learning models to predict diabetes and heart disease risk. 
          Get instant health insights based on your medical parameters.
        </p>
        <div style={{ display: 'flex', gap: '20px', justifyContent: 'center', flexWrap: 'wrap' }}>
          <Link to="/diabetes" className="btn">
            🔍 Check Diabetes Risk
          </Link>
          <Link to="/heart-disease" className="btn btn-secondary">
            ❤️ Check Heart Disease Risk
          </Link>
          <Link to="/general-disease" className="btn btn-tertiary">
            🧩 Check General Disease
          </Link>
        </div>
      </div>

      <div className="features-grid">
        <div className="feature-card">
          <div className="feature-icon">🤖</div>
          <h3>AI-Powered Predictions</h3>
          <p>
            Our advanced machine learning models analyze your health data to provide 
            accurate risk assessments for diabetes and heart disease.
          </p>
        </div>

        <div className="feature-card">
          <div className="feature-icon">⚡</div>
          <h3>Instant Results</h3>
          <p>
            Get your health risk assessment in seconds. No waiting, no appointments - 
            just enter your data and receive immediate insights.
          </p>
        </div>

        <div className="feature-card">
          <div className="feature-icon">📊</div>
          <h3>Detailed Analysis</h3>
          <p>
            Receive comprehensive risk assessments with probability scores and 
            personalized recommendations based on your health profile.
          </p>
        </div>

        <div className="feature-card">
          <div className="feature-icon">🔒</div>
          <h3>Privacy First</h3>
          <p>
            Your health data is processed securely and never stored. 
            We prioritize your privacy and data protection.
          </p>
        </div>

        <div className="feature-card">
          <div className="feature-icon">📱</div>
          <h3>Easy to Use</h3>
          <p>
            Simple, intuitive interface designed for everyone. 
            No medical expertise required to get your health insights.
          </p>
        </div>

        <div className="feature-card">
          <div className="feature-icon">🎯</div>
          <h3>Evidence-Based</h3>
          <p>
            Built on validated medical datasets and peer-reviewed research. 
            Our models are trained on real-world health data.
          </p>
        </div>
      </div>

      <div className="card">
        <h2>How It Works</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '30px', marginTop: '30px' }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '3rem', marginBottom: '15px' }}>1️⃣</div>
            <h3>Enter Your Data</h3>
            <p>Provide your health parameters like age, blood pressure, glucose levels, and other relevant metrics.</p>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '3rem', marginBottom: '15px' }}>2️⃣</div>
            <h3>AI Analysis</h3>
            <p>Our machine learning models analyze your data using advanced algorithms trained on medical datasets.</p>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '3rem', marginBottom: '15px' }}>3️⃣</div>
            <h3>Get Results</h3>
            <p>Receive your personalized risk assessment with detailed probability scores and recommendations.</p>
          </div>
        </div>
      </div>

      <div className="card">
        <h2>Important Disclaimer</h2>
        <div className="alert alert-info">
          <strong>⚠️ Medical Disclaimer:</strong> This application is for educational and demonstration purposes only. 
          It should not be used as a substitute for professional medical advice, diagnosis, or treatment. 
          Always consult with qualified healthcare professionals for medical concerns.
        </div>
        <p style={{ marginTop: '20px', textAlign: 'center' }}>
          The predictions provided by this system are based on machine learning models and should be used 
          as a screening tool only. They do not constitute medical diagnosis or treatment recommendations.
        </p>
      </div>
    </div>
  );
};

export default Home;
