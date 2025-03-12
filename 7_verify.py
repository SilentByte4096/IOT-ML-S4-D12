import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load the test dataset
test_data = pd.read_csv("classified_results.csv")  
print("Columns in test_data:", test_data.columns)  # Debugging step

# Check for required columns
if "Modulation" in test_data.columns and "Predicted Modulation" in test_data.columns:
    y_test = test_data["Modulation"]  # Actual labels
    y_pred = test_data["Predicted Modulation"]  # Model's predictions
else:
    raise ValueError("Error: Required columns ('Modulation', 'Predicted Modulation') are missing. Check CSV format.")

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Data: {accuracy:.2%}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
