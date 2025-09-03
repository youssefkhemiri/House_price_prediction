@echo off
REM House Price Prediction Crawler - Windows Startup Scripts
REM Run this script to start the Celery crawler system

echo.
echo ============================================================
echo    House Price Prediction - Celery Crawler System
echo ============================================================
echo.

REM Check if we're in the correct directory
if not exist "scripts\crawlers\celery_timer.py" (
    echo ERROR: Please run this script from the project root directory
    echo Current directory: %CD%
    echo Expected files: scripts\crawlers\celery_timer.py
    pause
    exit /b 1
)

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Install dependencies if needed
echo Checking dependencies...
pip show celery >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Celery dependencies...
    pip install -r requirements-celery.txt
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

echo Dependencies are ready!
echo.

REM Create necessary directories
if not exist "data\raw" mkdir "data\raw"
if not exist "data\raw\backups" mkdir "data\raw\backups"
if not exist "logs" mkdir "logs"

echo Directories created successfully!
echo.

REM Display menu
:MENU
echo ============================================================
echo Select an option:
echo ============================================================
echo 1. Install and setup everything (run once)
echo 2. Start Celery Worker (run in terminal #1)
echo 3. Start Celery Beat Scheduler (run in terminal #2) 
echo 4. Start Celery Flower Monitor (optional)
echo 5. Run Menzili crawler now (manual)
echo 6. Run Mubawab crawler now (manual)
echo 7. Run both crawlers now (manual)
echo 8. Check system status
echo 9. View recent logs
echo 0. Exit
echo ============================================================

set /p choice="Enter your choice (0-9): "

if "%choice%"=="1" goto SETUP
if "%choice%"=="2" goto WORKER
if "%choice%"=="3" goto BEAT
if "%choice%"=="4" goto FLOWER
if "%choice%"=="5" goto RUN_MENZILI
if "%choice%"=="6" goto RUN_MUBAWAB
if "%choice%"=="7" goto RUN_BOTH
if "%choice%"=="8" goto STATUS
if "%choice%"=="9" goto LOGS
if "%choice%"=="0" goto EXIT

echo Invalid choice. Please try again.
echo.
goto MENU

:SETUP
echo.
echo Running setup script...
python scripts\crawlers\setup_celery.py
echo.
echo Setup completed! 
echo IMPORTANT: Make sure Redis server is running before starting workers.
echo You can download Redis from: https://redis.io/download
echo Or run with Docker: docker run -d -p 6379:6379 redis:latest
echo.
pause
goto MENU

:WORKER
echo.
echo Starting Celery Worker...
echo Keep this terminal window open
echo Press Ctrl+C to stop the worker
echo.
celery -A scripts.crawlers.celery_timer worker --loglevel=info --concurrency=2 --pool=solo
goto MENU

:BEAT
echo.
echo Starting Celery Beat Scheduler...
echo Keep this terminal window open
echo Press Ctrl+C to stop the scheduler
echo.
celery -A scripts.crawlers.celery_timer beat --loglevel=info
goto MENU

:FLOWER
echo.
echo Starting Celery Flower Monitor...
echo Open http://localhost:5555 in your browser
echo Press Ctrl+C to stop
echo.
celery -A scripts.crawlers.celery_timer flower --port=5555
goto MENU

:RUN_MENZILI
echo.
echo Running Menzili crawler manually...
celery -A scripts.crawlers.celery_timer call celery_timer.run_menzili_now
echo.
echo Task submitted! Check worker logs for progress.
echo.
pause
goto MENU

:RUN_MUBAWAB
echo.
echo Running Mubawab crawler manually...
celery -A scripts.crawlers.celery_timer call celery_timer.run_mubawab_now
echo.
echo Task submitted! Check worker logs for progress.
echo.
pause
goto MENU

:RUN_BOTH
echo.
echo Running both crawlers manually...
celery -A scripts.crawlers.celery_timer call celery_timer.run_both_crawlers_now
echo.
echo Task submitted! Check worker logs for progress.
echo.
pause
goto MENU

:STATUS
echo.
echo ============================================================
echo                    SYSTEM STATUS
echo ============================================================
echo.

REM Check Python
echo Python Version:
python --version
echo.

REM Check Celery
echo Celery Status:
celery --version
echo.

REM Check Redis connection
echo Redis Connection:
python -c "import redis; r = redis.Redis(); r.ping(); print('✅ Redis connection: OK')" 2>nul || echo "❌ Redis connection: FAILED"
echo.

REM Check data files
echo Data Files:
if exist "data\raw\menzili_listings.json" (
    echo ✅ Menzili data file exists
) else (
    echo ❌ Menzili data file missing
)

if exist "data\raw\mubawab_listings.json" (
    echo ✅ Mubawab data file exists  
) else (
    echo ❌ Mubawab data file missing
)
echo.

REM Check recent activity
echo Recent Activity:
if exist "logs\daily_summary_*.json" (
    echo ✅ Daily summary logs found
) else (
    echo ❌ No daily summary logs found
)
echo.

pause
goto MENU

:LOGS
echo.
echo ============================================================
echo                    RECENT LOGS
echo ============================================================
echo.

REM Show recent task logs
if exist "logs\menzili_tasks.jsonl" (
    echo Last Menzili task:
    powershell "Get-Content 'logs\menzili_tasks.jsonl' | Select-Object -Last 1"
    echo.
)

if exist "logs\mubawab_tasks.jsonl" (
    echo Last Mubawab task:
    powershell "Get-Content 'logs\mubawab_tasks.jsonl' | Select-Object -Last 1"
    echo.
)

REM Show recent daily summaries
echo Recent daily summaries:
for %%f in (logs\daily_summary_*.json) do (
    echo Found: %%f
)
echo.

pause
goto MENU

:EXIT
echo.
echo Goodbye!
echo.
exit /b 0
