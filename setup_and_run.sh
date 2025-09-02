#!/bin/bash

# 🚀 Digital DeepFake AI Detection System - Setup & Run Script
# Production-ready automated setup for Linux/macOS

echo "🚀 Digital DeepFake AI Detection System - Setup & Run"
echo "======================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3.8+ is required but not installed."
    echo "Please install Python 3.8+ and try again."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js 16+ is required but not installed."
    echo "Please install Node.js 16+ and try again."
    exit 1
fi

echo "✅ Prerequisites check passed"
echo ""

# Setup Backend
echo "🔧 Setting up Backend..."
cd backend

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
mkdir -p ../uploads ../reports ../temp ../demo_datasets

echo "✅ Backend setup complete"
echo ""

# Setup Frontend
echo "🔧 Setting up Frontend..."
cd ../frontend

# Install Node.js dependencies
echo "Installing Node.js dependencies..."
npm install

echo "✅ Frontend setup complete"
echo ""

# Start services
echo "🚀 Starting services..."
echo ""

# Start backend in background
echo "Starting backend server on http://localhost:8000"
cd ../backend
source venv/bin/activate
nohup python enhanced_detection_api.py > backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start frontend
echo "Starting frontend server on http://localhost:3000"
cd ../frontend
npm start &
FRONTEND_PID=$!

echo ""
echo "🎉 System is now running!"
echo "======================================================"
echo "Frontend: http://localhost:3000"
echo "Backend API: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo ""
echo "Demo Pages:"
echo "- Interactive Demo: http://localhost:3000/demo"
echo "- Workflow: http://localhost:3000/workflow"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for user to stop services
trap "echo 'Stopping services...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait
