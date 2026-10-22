@echo off
rem MorpHista: double-click to start. The first start installs what the program needs (5-20 minutes,
rem internet required); later starts open the window straight away.
setlocal
cd /d "%~dp0"
title MorpHista

set "PYW=%~dp0.pixi\envs\default\pythonw.exe"
if exist "%PYW%" goto :start
call :findpixi

echo.
echo  MorpHista - first start
echo  =======================
echo  The program needs Python, PyTorch and a few other packages. They are downloaded once into this
echo  folder (about 5 GB to download, 6 GB on disk). This takes 5-20 minutes. Please keep this window open.
echo.

if defined PIXI goto :packages
echo  [1/2] Installing the package manager (pixi)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm -UseBasicParsing https://pixi.sh/install.ps1 | iex"
call :findpixi
if not defined PIXI goto :failed

:packages
echo.
echo  [2/2] Installing Python and the program's packages...
"%PIXI%" install --frozen
if errorlevel 1 goto :failed
if not exist "%PYW%" goto :failed

echo.
echo  Creating a "MorpHista" shortcut on the desktop...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$s = (New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop') + '\MorpHista.lnk');" ^
  "$s.TargetPath = '%~dp0Start MorpHista.cmd'; $s.WorkingDirectory = '%~dp0';" ^
  "$s.IconLocation = '%~dp0assets\icon.ico'; $s.WindowStyle = 7; $s.Save()"
echo  Done. From now on, start the program from the desktop shortcut.
echo.

:start
start "" "%PYW%" app.py
exit /b 0

:findpixi
rem where the pixi installer puts it (newer and older versions), or anywhere on PATH
set "PIXI="
if exist "%LOCALAPPDATA%\pixi\bin\pixi.exe" set "PIXI=%LOCALAPPDATA%\pixi\bin\pixi.exe"
if not defined PIXI if exist "%USERPROFILE%\.pixi\bin\pixi.exe" set "PIXI=%USERPROFILE%\.pixi\bin\pixi.exe"
if not defined PIXI for /f "delims=" %%p in ('where pixi 2^>nul') do if not defined PIXI set "PIXI=%%p"
exit /b 0

:failed
echo.
echo  Installation did not finish. Check the internet connection and start this file again.
echo  If it keeps failing, send the text in this window to the author.
echo.
pause
exit /b 1
