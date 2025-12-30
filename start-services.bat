@echo off

REM Kill any processes using project ports
echo Cleaning up existing processes...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
    echo Killing Django process on port 8000 with PID %%a
    taskkill /PID %%a /F
)

@REM netstat -ano | findstr :8000
@REM taskkill /PID 3972 /F

echo Starting FYP Crypto Data Ingestion Services...

cd src

REM Start Celery Worker for maintenance tasks only
echo Starting Celery Worker for maintenance tasks...
start "Celery Maintenance" cmd /k celery -A configuration worker --loglevel=info --concurrency=1 -Q maintenance

REM Start Celery Beat for scheduled tasks
echo Starting Celery Beat scheduler...
start "Celery Beat Scheduler" cmd /k celery -A configuration beat --loglevel=info --scheduler django_celery_beat.schedulers:DatabaseScheduler

REM Start Flower for task monitoring
echo Starting Flower monitoring...
start "Flower Monitor" cmd /k celery -A configuration flower --port=5555 --broker=redis://localhost:6379/0

REM Wait a moment for services to initialize
timeout /t 5 /nobreak > nul

REM Start multiple trading orchestrators for different shards
echo Starting trading orchestrators...
start "Trading Orchestrator Shard 0" cmd /k python manage.py run_trading_orchestrator --shard-name=shard_0 --phase=both
timeout /t 2 /nobreak > nul
start "Trading Orchestrator Shard 1" cmd /k python manage.py run_trading_orchestrator --shard-name=shard_1 --phase=both
timeout /t 2 /nobreak > nul
start "Trading Orchestrator Shard 2" cmd /k python manage.py run_trading_orchestrator --shard-name=shard_2 --phase=both

REM Start Django server
echo Starting Django server...
start "Django Server" cmd /k python manage.py runserver 0.0.0.0:8000 --noreload

echo.
echo ========================================
echo All FYP Crypto Services Started!
echo ========================================
echo Django Admin: http://localhost:8000/admin/
echo Flower Monitor: http://localhost:5555/
echo InfluxDB UI: http://localhost:8086/
echo.
echo Services running:
echo - PostgreSQL, InfluxDB, Redis (Infrastructure)
echo - Trading Orchestrators (3 shards)
echo - Celery Worker + Beat (Maintenance only)
echo - Django Server
echo.
echo Management Commands:
echo - Check status: python manage.py run_trading_orchestrator --status
echo - List shards: python manage.py run_trading_orchestrator --list-shards
echo - Force reinit: python manage.py init_trading --reset --workers=3
echo.
echo Press any key to show service status...
pause > nul
docker-compose ps
echo.
echo To stop: docker-compose down