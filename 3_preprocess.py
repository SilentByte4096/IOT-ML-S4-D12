import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("features.csv")  # Ensure your dataset is named correctly

# Normalize features
scaler = StandardScaler()
feature_columns = ["Signal Power", "FFT Mean", "SNR"]  # Adjust as needed
df[feature_columns] = scaler.fit_transform(df[feature_columns])

# Save preprocessed data
df.to_csv("processed_features.csv", index=False)
print("Preprocessing complete. Saved as processed_features.csv")
