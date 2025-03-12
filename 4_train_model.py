import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("features.csv")  # Ensure this file exists

# Extract features and labels
X = df.iloc[:, :-1].values  # Features
y = df.iloc[:, -1].values   # Labels

# Standardize features (helps generalization)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Increase test size (more real-world data for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42, stratify=y)

# Apply noise augmentation (for robustness)
X_train += np.random.normal(0, 0.1, X_train.shape)  # Stronger noise

# Use SVM (better for small datasets & avoids memorization)
model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)

# Cross-validation accuracy
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"Cross-validation accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

# Train the model
model.fit(X_train, y_train)

# Evaluate on test set
y_pred = model.predict(X_test)
print("\nModel Performance on Test Data:")
print(classification_report(y_test, y_pred))

# Save model & scaler
joblib.dump(model, "modulation_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved!")