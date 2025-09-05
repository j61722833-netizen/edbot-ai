@echo off
REM EdBot AI Runner Script for Windows
REM Automatically activates virtual environment and runs CLI

if not exist "venv" (
    echo ❌ Virtual environment not found!
    echo Please run: python -m venv venv ^&^& venv\Scripts\activate ^&^& pip install -r requirements.txt
    exit /b 1
)

REM Activate virtual environment and run CLI
call venv\Scripts\activate
python cli.py %*