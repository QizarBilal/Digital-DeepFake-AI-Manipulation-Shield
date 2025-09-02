@echo off
echo ====================================
echo  Enhanced AI Detection System v3.0
echo ====================================
echo.
echo This system provides:
echo - Multi-format AI detection (text, image, video, audio)
echo - Interactive demo visualization
echo - Workflow presentation
echo - Production-ready API
echo - Professional dark theme UI
echo.
echo Starting both frontend and backend servers...
echo.

:: Start backend in a new window
echo Starting backend server...
start "AI Detection Backend" /d "E:\Resume Projects\Digital DeepFake AI\deepfake-detection-system" start_backend.bat

:: Wait a moment for backend to start
timeout /t 3 /nobreak > nul

:: Start frontend in a new window
echo Starting frontend server...
start "AI Detection Frontend" /d "E:\Resume Projects\Digital DeepFake AI\deepfake-detection-system" start_frontend.bat

echo.
echo Both servers are starting in separate windows...
echo.
echo Backend: http://localhost:8000 (API Documentation: /docs)
echo Frontend: http://localhost:3000
echo.
echo Demo Pages:
echo - Interactive Demo: http://localhost:3000/demo
echo - Workflow: http://localhost:3000/workflow
echo.
echo Press any key to exit this launcher...
pause > nul
