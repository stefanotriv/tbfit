@echo off
rem Starts the tbfit GUI. Installs numpy, scipy and matplotlib first if they are missing.
setlocal EnableDelayedExpansion
cd /d "%~dp0"

rem ---- find Python 3 (an activated conda/venv "python" wins, then the py launcher) ----
set "PY="
python -c "import sys; sys.exit(sys.version_info[0] < 3)" >nul 2>&1 && set "PY=python"
if not defined PY py -3 -c "import sys" >nul 2>&1 && set "PY=py -3"
if not defined PY goto :nopython

for /f "delims=" %%V in ('%PY% -c "import sys; print(sys.version.split()[0])"') do set "PYVER=%%V"
echo Using Python %PYVER% ^(%PY%^)

rem ---- tkinter is part of Python and cannot be installed with pip ----
%PY% -c "import tkinter" >nul 2>&1
if errorlevel 1 goto :notk

rem ---- install missing packages ----
set "MISSING="
for %%P in (numpy scipy matplotlib) do (
    %PY% -c "import %%P" >nul 2>&1 || set "MISSING=!MISSING! %%P"
)
if not defined MISSING goto :run

echo Missing packages:%MISSING%
%PY% -m pip --version >nul 2>&1 || %PY% -m ensurepip --upgrade
%PY% -m pip install%MISSING%
if errorlevel 1 goto :pipfail

:run
echo Starting tbgui...
%PY% tbgui.py %*
if errorlevel 1 (
    echo.
    echo tbgui exited with an error, see the messages above.
    pause
    exit /b 1
)
exit /b 0

:nopython
echo Python 3 was not found.
echo Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH",
echo then run this file again.
pause
exit /b 1

:notk
echo This Python has no tkinter. Reinstall Python from https://www.python.org/downloads/
echo and keep the "tcl/tk and IDLE" option ticked.
pause
exit /b 1

:pipfail
echo.
echo Installing%MISSING% failed. Try manually:  %PY% -m pip install --user%MISSING%
pause
exit /b 1
