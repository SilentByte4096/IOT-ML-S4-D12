import pandas as pd
import joblib

# Load trained model
model = joblib.load("modulation_model.pkl")

# Load test data
df = pd.read_csv("processed_features.csv")

# Predict on new data
X = df[["Signal Power", "FFT Mean", "SNR"]]
predictions = model.predict(X)

# Add predictions to DataFrame
df["Predicted Modulation"] = predictions

# Save results
df.to_csv("classified_results.csv", index=False)
print("Predictions saved to classified_results.csv")