#!/bin/bash

# Modulation Classification - Full Pipeline 
# This script automates the execution of the entire modulation classification pipeline.

# Ensure the script is run from the directory where it is located
echo "Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
echo "Installing dependencies..."
pip install -r requirements.txt

# Signal generation
echo "Step 1: Generating modulated signals..."
python3 1_generate_signal.py

#Feature extraction
echo "Step 2: Extracting features..."
python3 2_feature_extraction.py

# Training and testing ML models
echo "Step 3: Training and testing ML models..."
python3 3_train_test_model.py

# Real-time monitoring and prediction
echo "Step 4: Running real-time monitoring and prediction (15s window)..."
python3 4_real_time_15seconds.py

# Dashboard for visualization
echo "Step 5: Launching dashboard for visualization..."
python3 5_dashboard.py

echo "Pipeline execution complete."
