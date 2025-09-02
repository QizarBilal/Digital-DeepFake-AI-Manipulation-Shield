@echo off
echo Starting Enhanced AI Detection System...
echo.

:: Navigate to backend directory
cd /d "E:\Resume Projects\Digital DeepFake AI\deepfake-detection-system\backend"

:: Activate virtual environment and install dependencies
echo Activating virtual environment...
call "E:\Resume Projects\Digital DeepFake AI\.venv\Scripts\activate.bat"

echo Installing/updating dependencies...
pip install fastapi uvicorn python-multipart aiofiles Pillow numpy pandas scikit-learn --quiet

echo.
echo Starting backend server...
echo Backend will be available at: http://localhost:8000
echo API documentation at: http://localhost:8000/docs
echo.

:: Start the backend server
python enhanced_detection_api.py

pause
