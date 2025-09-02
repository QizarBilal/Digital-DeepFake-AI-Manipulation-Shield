@echo off
echo Starting Enhanced AI Detection System Frontend...
echo.

:: Navigate to frontend directory
cd /d "E:\Resume Projects\Digital DeepFake AI\deepfake-detection-system\frontend"

:: Install dependencies if needed
echo Installing/updating dependencies...
npm install --silent

echo.
echo Starting frontend development server...
echo Frontend will be available at: http://localhost:3000
echo.

:: Start the frontend server
npm start

pause
