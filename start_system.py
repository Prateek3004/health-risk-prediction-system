#!/usr/bin/env python3
"""
Startup Script for Health Risk Prediction System

This script provides an easy way to start the entire system including:
1. Training models (if needed)
2. Starting the Flask backend
3. Starting the React frontend
"""

import os
import sys
import subprocess
import time
import threading
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    # Check Python dependencies
    try:
        import flask
        import pandas
        import sklearn
        import joblib
        print("✅ Python dependencies are installed")
    except ImportError as e:
        print(f"❌ Missing Python dependency: {e}")
        print("Please run: pip install -r backend/requirements.txt")
        return False
    
    # Check if Node.js is installed
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Node.js is installed: {result.stdout.strip()}")
        else:
            print("❌ Node.js is not installed")
            return False
    except FileNotFoundError:
        print("❌ Node.js is not installed")
        return False
    
    # Check if npm is installed
    try:
        result = subprocess.run(['npm', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ npm is installed: {result.stdout.strip()}")
        else:
            print("❌ npm is not installed")
            return False
    except FileNotFoundError:
        print("❌ npm is not installed")
        return False
    
    return True

def check_models():
    """Check if trained models exist"""
    model_files = [
        "models/diabetes_lr.pkl",
        "models/diabetes_scaler.pkl",
        "models/heart_disease_lr.pkl",
        "models/heart_disease_scaler.pkl"
    ]
    
    missing_models = []
    for model_file in model_files:
        if not os.path.exists(model_file):
            missing_models.append(model_file)
    
    if missing_models:
        print("⚠️  Missing model files:")
        for model in missing_models:
            print(f"   - {model}")
        return False
    else:
        print("✅ All model files found")
        return True

def train_models():
    """Train the models if they don't exist"""
    print("\n🔄 Training models...")
    try:
        result = subprocess.run([sys.executable, "train_models.py"], 
                              cwd=os.getcwd(), check=True)
        print("✅ Models trained successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error training models: {e}")
        return False

def install_frontend_dependencies():
    """Install frontend dependencies"""
    print("\n📦 Installing frontend dependencies...")
    try:
        result = subprocess.run(['npm', 'install'], 
                              cwd='frontend', check=True)
        print("✅ Frontend dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing frontend dependencies: {e}")
        return False

def start_backend():
    """Start the Flask backend server"""
    print("\n🚀 Starting Flask backend server...")
    try:
        # Change to backend directory and start Flask
        backend_process = subprocess.Popen(
            [sys.executable, "app.py"],
            cwd="backend",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a moment for the server to start
        time.sleep(3)
        
        # Check if the process is still running
        if backend_process.poll() is None:
            print("✅ Backend server started on http://localhost:5000")
            return backend_process
        else:
            stdout, stderr = backend_process.communicate()
            print(f"❌ Backend server failed to start:")
            print(f"   stdout: {stdout.decode()}")
            print(f"   stderr: {stderr.decode()}")
            return None
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return None

def start_frontend():
    """Start the React frontend server"""
    print("\n🚀 Starting React frontend server...")
    try:
        # Change to frontend directory and start React
        frontend_process = subprocess.Popen(
            ['npm', 'start'],
            cwd="frontend",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a moment for the server to start
        time.sleep(5)
        
        # Check if the process is still running
        if frontend_process.poll() is None:
            print("✅ Frontend server started on http://localhost:3000")
            return frontend_process
        else:
            stdout, stderr = frontend_process.communicate()
            print(f"❌ Frontend server failed to start:")
            print(f"   stdout: {stdout.decode()}")
            print(f"   stderr: {stderr.decode()}")
            return None
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")
        return None

def main():
    """Main function to start the entire system"""
    print("🏥 Health Risk Prediction System - Startup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("data") or not os.path.exists("backend") or not os.path.exists("frontend"):
        print("❌ Please run this script from the health-risk-predictor directory")
        return
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again")
        return
    
    # Check models
    if not check_models():
        print("\n🔄 Models not found. Training new models...")
        if not train_models():
            print("❌ Failed to train models. Please check the data files.")
            return
    
    # Install frontend dependencies if needed
    if not os.path.exists("frontend/node_modules"):
        if not install_frontend_dependencies():
            print("❌ Failed to install frontend dependencies")
            return
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        print("❌ Failed to start backend server")
        return
    
    # Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        print("❌ Failed to start frontend server")
        backend_process.terminate()
        return
    
    print("\n" + "=" * 50)
    print("🎉 Health Risk Prediction System is running!")
    print("=" * 50)
    print("📱 Frontend: http://localhost:3000")
    print("🔌 Backend API: http://localhost:5000")
    print("📊 API Health Check: http://localhost:5000/health")
    print("\n💡 Press Ctrl+C to stop the servers")
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if backend_process.poll() is not None:
                print("❌ Backend server stopped unexpectedly")
                break
            if frontend_process.poll() is not None:
                print("❌ Frontend server stopped unexpectedly")
                break
                
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down servers...")
        
        # Terminate processes
        if backend_process:
            backend_process.terminate()
            print("✅ Backend server stopped")
        
        if frontend_process:
            frontend_process.terminate()
            print("✅ Frontend server stopped")
        
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()
