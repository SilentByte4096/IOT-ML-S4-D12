import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Define modulation types
modulation_types = ["BPSK", "QPSK", "16-QAM"]

# Create output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Load dataset
data_file = "features_with_modulation.csv"
df = pd.read_csv(data_file)

# Features and target
feature_columns = ["mean", "variance", "skewness", "kurtosis", "fft_mean", "spectral_entropy", 
                   "signal_power", "energy", "snr_db"]
X = df[feature_columns]
y = df["modulation_type"]

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, random_state=42)  # 500 test samples

# Initialize models with regularization
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=5, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42),
    "SVM": SVC(kernel='rbf', C=0.5, gamma='scale', probability=True, random_state=42)
}

# Train and evaluate models with cross-validation
print("Training and Cross-Validation Scores:")
for name, model in models.items():
    model.fit(X_train, y_train)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{name}: Mean CV Accuracy = {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Predict with each model on test set
predictions = {name: model.predict(X_test) for name, model in models.items()}

# Majority voting for test set
y_pred_majority = []
for i in range(len(X_test)):
    votes = [predictions["RandomForest"][i], predictions["GradientBoosting"][i], predictions["SVM"][i]]
    unique, counts = np.unique(votes, return_counts=True)
    majority_vote = unique[np.argmax(counts)]
    y_pred_majority.append(majority_vote)
y_pred_majority = np.array(y_pred_majority)

# Evaluate majority voting
accuracy = accuracy_score(y_test, y_pred_majority)
print(f"\nMajority Voting Accuracy: {accuracy:.4f}")
print("Classification Report (Majority Voting):")
print(classification_report(y_test, y_pred_majority, target_names=modulation_types))

# Save models to output folder
for name, model in models.items():
    model_path = os.path.join(output_dir, f"modulation_predictor_{name}.pkl")
    joblib.dump(model, model_path)
    print(f"{name} model saved to {model_path}")

# Predict on entire dataset with majority voting
all_preds = {name: model.predict(X) for name, model in models.items()}
df["rf_pred"] = all_preds["RandomForest"]
df["gb_pred"] = all_preds["GradientBoosting"]
df["svm_pred"] = all_preds["SVM"]
df["predicted_modulation"] = [np.unique([all_preds["RandomForest"][i], all_preds["GradientBoosting"][i], all_preds["SVM"][i]], return_counts=True)[0][np.argmax(np.unique([all_preds["RandomForest"][i], all_preds["GradientBoosting"][i], all_preds["SVM"][i]], return_counts=True)[1])] for i in range(len(X))]

# Save predictions to output folder
csv_path = os.path.join(output_dir, "features_with_predictions.csv")
df.to_csv(csv_path, index=False)
print(f"Predictions saved to {csv_path}")

# Print one random example from test set with feature names preserved
rng = np.random.RandomState(42)
random_idx = rng.randint(0, len(X_test))
sample = pd.DataFrame([X_test.iloc[random_idx]], columns=feature_columns)  # Preserve feature names
rf_pred = models["RandomForest"].predict(sample)[0]
gb_pred = models["GradientBoosting"].predict(sample)[0]
svm_pred = models["SVM"].predict(sample)[0]
unique, counts = np.unique([rf_pred, gb_pred, svm_pred], return_counts=True)
majority_pred = unique[np.argmax(counts)]
true_mod = y_test.iloc[random_idx]

print(f"\nRandom Example (Test Sample {random_idx}):")
print(f"Features: {X_test.iloc[random_idx].to_dict()}")
print(f"RandomForest Prediction: {rf_pred}")
print(f"GradientBoosting Prediction: {gb_pred}")
print(f"SVM Prediction: {svm_pred}")
print(f"Majority Voting Prediction: {majority_pred}")
print(f"True Modulation: {true_mod}")