@echo off
echo ==========================================
echo Starting Parkinson's Screening Pipeline
echo ==========================================

echo [1/3] Extracting Features from All Videos...
echo This operation may take a long time. Please wait.
python build_feature_csv.py
if %ERRORLEVEL% NEQ 0 (
    echo Error extracting features. Exiting.
    pause
    exit /b %ERRORLEVEL%
)

echo [2/3] Training Model...
python train_model.py
if %ERRORLEVEL% NEQ 0 (
    echo Error training model. Exiting.
    pause
    exit /b %ERRORLEVEL%
)

echo [3/3] Running Prediction on Videos...
python predict_folder.py
if %ERRORLEVEL% NEQ 0 (
    echo Error running prediction. Exiting.
    pause
    exit /b %ERRORLEVEL%
)

echo ==========================================
echo Pipeline Completed Successfully!
echo ==========================================
pause
