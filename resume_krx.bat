@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist logs mkdir logs
echo ================================================== >> logs\resume_krx.log
echo [%date% %time%] KRX 차단 해제 확인 >> logs\resume_krx.log
python scripts\10_resume_krx.py >> logs\resume_krx.log 2>&1
echo (exit %errorlevel%) >> logs\resume_krx.log
