@echo off
title Architect Local LLM Server & Chat UI
echo ========================================================
echo   Starting Architect Local LLM Server (FastAPI)
echo ========================================================
cd /d "%~dp0"

echo [*] Loading model weights into memory... Please wait...
start cmd /k "python server.py"

echo [*] Waiting 4 seconds for the server to spin up...
timeout /t 4 >nul

echo [*] Opening Web Chat UI in browser...
start http://127.0.0.1:8000
echo [+] Server is running! Keep the server command window open.
pause
