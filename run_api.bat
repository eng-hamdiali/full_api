@echo off
cd /d %~dp0
call activate api_venv
set /p filename="Enter Python FastAPI file name (without extension): "
uvicorn %filename%:app  --reload
rem pp
pause
cmd