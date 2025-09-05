#!/bin/bash
# EdBot AI Runner Script
# Automatically activates virtual environment and runs CLI

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Load .env file if it exists (for environment variables)
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Activate virtual environment and run CLI
source venv/bin/activate
python cli.py "$@"