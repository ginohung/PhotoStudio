@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================
echo   PhotoStudio  一鍵啟動
echo ============================================

REM --- 選擇 Python 直譯器（優先 3.12，其次 3.11）---
set "PYEXE="
py -3.12 --version >nul 2>&1 && set "PYEXE=py -3.12"
if not defined PYEXE ( py -3.11 --version >nul 2>&1 && set "PYEXE=py -3.11" )

if not defined PYEXE (
    echo [警告] 找不到 Python 3.12 或 3.11。
    echo        rembg/onnxruntime 對 Python 3.14 尚無官方 wheel，安裝可能失敗。
    echo        請至 https://www.python.org/downloads/ 安裝 Python 3.12。
    echo.
    echo        仍要用預設 python 嘗試嗎？按任意鍵繼續，或關閉視窗中止。
    pause >nul
    set "PYEXE=python"
)

echo 使用直譯器：%PYEXE%

REM --- 建立虛擬環境（若不存在）---
if not exist ".venv\Scripts\python.exe" (
    echo 建立虛擬環境 .venv ...
    %PYEXE% -m venv .venv
    if errorlevel 1 (
        echo [錯誤] 建立虛擬環境失敗。
        pause
        exit /b 1
    )
)

REM --- 安裝相依套件 ---
echo 檢查 / 安裝相依套件 ...
call ".venv\Scripts\python.exe" -m pip install --upgrade pip
call ".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [錯誤] 套件安裝失敗，請檢查上方訊息。
    pause
    exit /b 1
)

REM --- 啟動 Streamlit ---
echo.
echo 啟動網站中，瀏覽器將自動開啟 http://localhost:8501 ...
echo （關閉本視窗即可停止服務）
echo.
call ".venv\Scripts\python.exe" -m streamlit run app.py

endlocal
