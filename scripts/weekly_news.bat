@echo off
rem 증권사 목표가 주간 수집 - Windows 작업 스케줄러가 주 1회 실행한다.
rem 작업 폴더는 스케줄러가 screener 로 지정한다(경로에 한글이 있어 여기 적지 않는다).
rem 대형주는 검색어당 1,000건 한도 때문에 한 번에 약 30일치만 온다.
rem 주 1회 쌓아야 90일 창이 채워진다. 월간 실행과 별개로 목표가만 받는다.
chcp 65001 > nul
if not exist logs mkdir logs
echo.>> logs\news_weekly.log
echo ===== %date% %time% =====>> logs\news_weekly.log
rem 파이썬 실행 파일 경로가 PATH 에 없으면 아래를 절대경로로 바꾼다
python scripts_news_consensus.py >> logs
ews_weekly.log 2>&1
exit /b %errorlevel%
