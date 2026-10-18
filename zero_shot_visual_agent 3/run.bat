@echo off
REM One-click launcher (Windows): creates a virtual env, installs dependencies, starts the app.
cd /d "%~dp0"
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv || (echo Python 3.10+ is required. Install it from python.org & pause & exit /b 1)
)
call .venv\Scripts\activate.bat
if not exist .venv\.installed (
    echo Installing dependencies ^(first time only, a few minutes^)...
    python -m pip install --upgrade pip
    pip install -r requirements.txt || (echo Install failed & pause & exit /b 1)
    echo done> .venv\.installed
)
echo Starting app at http://localhost:8501 ...
streamlit run app.py
pause
