@echo off
echo Building AudioTransfer...

pip install -r requirements.txt

pyinstaller --onefile --windowed --name "AudioTransfer" --icon "AudioTransfer.ico" --version-file "version_info.txt" --hidden-import soundcard --hidden-import soundcard._mediafoundation --hidden-import numpy --collect-all soundcard --add-data "AudioTransfer.ico;." main.py

echo.
echo Done! EXE is at dist\AudioTransfer.exe
pause
