import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import './App.css';
import DiabetesPredictor from './components/DiabetesPredictor';
import HeartDiseasePredictor from './components/HeartDiseasePredictor';
import GeneralDiseasePredictor from './components/GeneralDiseasePredictor';
import Home from './components/Home';
import Dashboard from './components/Dashboard';

function App() {
  return (
    <Router>
      <div className="App">
        <nav className="navbar">
          <div className="nav-container">
            <Link to="/" className="nav-logo">
              🏥 Health Risk Predictor
            </Link>
            <ul className="nav-menu">
              <li className="nav-item">
                <Link to="/" className="nav-link">Home</Link>
              </li>
              <li className="nav-item">
                <Link to="/diabetes" className="nav-link">Diabetes Risk</Link>
              </li>
              <li className="nav-item">
                <Link to="/heart-disease" className="nav-link">Heart Disease Risk</Link>
              </li>
              <li className="nav-item">
                <Link to="/general-disease" className="nav-link">General Disease</Link>
              </li>
              <li className="nav-item">
                <Link to="/dashboard" className="nav-link">Dashboard</Link>
              </li>
            </ul>
          </div>
        </nav>

        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/diabetes" element={<DiabetesPredictor />} />
            <Route path="/heart-disease" element={<HeartDiseasePredictor />} />
            <Route path="/general-disease" element={<GeneralDiseasePredictor />} />
            <Route path="/dashboard" element={<Dashboard />} />
          </Routes>
        </main>

        <footer className="footer">
          <p>&copy; 2024 Health Risk Prediction System. This is a demonstration project and should not be used for actual medical diagnosis.</p>
        </footer>
      </div>
    </Router>
  );
}

export default App;
