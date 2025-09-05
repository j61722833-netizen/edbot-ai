@echo off
REM EdBot AI Runner Script for Windows
REM Automatically activates virtual environment and runs CLI

if not exist "venv" (
    echo ❌ Virtual environment not found!
    echo Please run: python -m venv venv ^&^& venv\Scripts\activate ^&^& pip install -r requirements.txt
    exit /b 1
)

REM Load .env file if it exists (for environment variables)
if exist ".env" (
    for /f "usebackq delims=" %%a in (".env") do (
        for /f "tokens=1,2 delims==" %%b in ("%%a") do (
            if not "%%b"=="" if not "%%b:~0,1%"=="#" set "%%b=%%c"
        )
    )
)

REM Activate virtual environment and run CLI
call venv\Scripts\activate
python cli.py %*