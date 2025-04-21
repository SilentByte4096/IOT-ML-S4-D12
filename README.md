# Modulation Classification using Machine Learning  

This project classifies different modulation types (BPSK, QPSK, 16-QAM) using machine learning techniques.  
It extracts features from SDR (Software Defined Radio) signals, trains a model, and evaluates performance.  

## 📌 Features  
- Generates simulated signals that are suitable for different modulations  
- Extracts key features from signal data  
- Preprocesses data for training  
- Trains a machine learning model and creates an ensemble model
- Evaluates and tests model performance  
- Visualizes classification results  

## 📂 File Structure  
- `1_signal_generation.py` - Generates synthetic modulated signals  
- `2_feature_extraction.py` - Extracts signal features and builds dataset
- `3_train_test_model.py` - Trains and tests ML models    
- `4_real_time_15seconds.py` - Real-time monitoring and modulation decision  
- `5_dashboard.py` - Visualizes predictions and performance  
- `README.md` - Project documentation and instructions for setup and usage

## 📊 Dataset  
The project works with synthetic signal data saved in CSV format:  
- `features.csv` - Extracted features from signals  
- `processed_features.csv` - Normalized features for training  
- `classified_results.csv` - Model predictions  

## Output Structure 
- `output/` - Stores results and metrics
- `metadata/` - Config or support files (for signal data)

## 🛠 Dependencies  
See [requirements.txt](requirements.txt)  

## 🚀 Running the Project  
See [RUN_INSTRUCTIONS.md](RUN_INSTRUCTIONS.md)

