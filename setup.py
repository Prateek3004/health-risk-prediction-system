#!/usr/bin/env python3
"""
Setup Script for Health Risk Prediction System

This script automates the setup process for the health risk prediction system.
"""

import os
import sys
import subprocess
import platform

def print_banner():
    """Print the setup banner"""
    print("🏥 Health Risk Prediction System - Setup")
    print("=" * 50)
    print("This script will help you set up the complete system.")
    print("=" * 50)

def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_nodejs():
    """Check if Node.js is installed"""
    print("\n🔍 Checking Node.js...")
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Node.js {result.stdout.strip()} is installed")
            return True
        else:
            print("❌ Node.js is not properly installed")
            return False
    except FileNotFoundError:
        print("❌ Node.js is not installed")
        print("💡 Please install Node.js from https://nodejs.org/")
        return False

def check_npm():
    """Check if npm is installed"""
    print("\n🔍 Checking npm...")
    try:
        result = subprocess.run(['npm', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ npm {result.stdout.strip()} is installed")
            return True
        else:
            print("❌ npm is not properly installed")
            return False
    except FileNotFoundError:
        print("❌ npm is not installed")
        return False

def create_virtual_environment():
    """Create a Python virtual environment"""
    print("\n🐍 Creating Python virtual environment...")
    if os.path.exists("venv"):
        print("✅ Virtual environment already exists")
        return True
    
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def get_activate_command():
    """Get the appropriate activate command for the OS"""
    if platform.system() == "Windows":
        return "venv\\Scripts\\activate"
    else:
        return "source venv/bin/activate"

def install_backend_dependencies():
    """Install backend Python dependencies"""
    print("\n📦 Installing backend dependencies...")
    try:
        # Install requirements
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "backend/requirements.txt"], check=True)
        print("✅ Backend dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install backend dependencies: {e}")
        return False

def install_frontend_dependencies():
    """Install frontend Node.js dependencies"""
    print("\n📦 Installing frontend dependencies...")
    try:
        subprocess.run(['npm', 'install'], cwd="frontend", check=True)
        print("✅ Frontend dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install frontend dependencies: {e}")
        return False

def train_models():
    """Train the machine learning models"""
    print("\n🤖 Training machine learning models...")
    try:
        subprocess.run([sys.executable, "train_models.py"], check=True)
        print("✅ Models trained successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to train models: {e}")
        return False

def create_startup_scripts():
    """Create startup scripts for different platforms"""
    print("\n📝 Creating startup scripts...")
    
    # Windows batch file
    if platform.system() == "Windows":
        with open("start_system.bat", "w") as f:
            f.write("@echo off\n")
            f.write("echo Starting Health Risk Prediction System...\n")
            f.write("venv\\Scripts\\activate\n")
            f.write("python start_system.py\n")
            f.write("pause\n")
        print("✅ Created start_system.bat for Windows")
    
    # Unix shell script
    else:
        with open("start_system.sh", "w") as f:
            f.write("#!/bin/bash\n")
            f.write("echo 'Starting Health Risk Prediction System...'\n")
            f.write("source venv/bin/activate\n")
            f.write("python start_system.py\n")
        
        # Make it executable
        os.chmod("start_system.sh", 0o755)
        print("✅ Created start_system.sh for Unix/Linux/macOS")

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 50)
    print("🎉 Setup Complete!")
    print("=" * 50)
    print("\n📋 Next Steps:")
    print("1. Activate the virtual environment:")
    
    if platform.system() == "Windows":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("\n2. Start the system:")
    if platform.system() == "Windows":
        print("   python start_system.py")
        print("   or double-click start_system.bat")
    else:
        print("   python start_system.py")
        print("   or ./start_system.sh")
    
    print("\n3. Open your browser and go to:")
    print("   http://localhost:3000")
    
    print("\n📚 For more information, see README.md")
    print("\n⚠️  Remember: This is for educational purposes only!")
    print("   Always consult healthcare professionals for medical advice.")

def main():
    """Main setup function"""
    print_banner()
    
    # Check prerequisites
    if not check_python_version():
        return
    
    if not check_nodejs():
        return
    
    if not check_npm():
        return
    
    # Create virtual environment
    if not create_virtual_environment():
        return
    
    # Install dependencies
    if not install_backend_dependencies():
        return
    
    if not install_frontend_dependencies():
        return
    
    # Train models
    if not train_models():
        return
    
    # Create startup scripts
    create_startup_scripts()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()
