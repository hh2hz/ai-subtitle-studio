@echo off
REM Starts AI Subtitle Studio from this folder using its virtual environment.
REM Create the environment once (see README.md):
REM   py -3 -m venv .venv
REM   .venv\Scripts\python.exe -m pip install -r requirements-dev.txt
setlocal
set "HERE=%~dp0"
set "PYTHONW=%HERE%.venv\Scripts\pythonw.exe"
if not exist "%PYTHONW%" (
    echo The virtual environment is missing: %PYTHONW%
    echo Create it first:  py -3 -m venv .venv  ^&^&  .venv\Scripts\python.exe -m pip install -r requirements-dev.txt
    pause
    exit /b 1
)
REM The app repairs and updates its libraries by itself at start-up; this only covers the case where even the
REM window library (PySide6) is missing, so that nothing ever has to be typed by hand.
"%HERE%.venv\Scripts\python.exe" -c "import PySide6" >nul 2>&1
if errorlevel 1 (
    echo Installing the required libraries, please wait...
    "%HERE%.venv\Scripts\python.exe" -m pip install -q -r "%HERE%requirements.txt"
    if errorlevel 1 (
        echo Installing the libraries failed. Check the internet connection and start launch.bat again.
        pause
        exit /b 1
    )
)
set "PYTHONPATH=%HERE%;%HERE%.venv\Lib\site-packages"
start "" "%PYTHONW%" -m app.main
endlocal
