import numpy as np
import pandas as pd
import scipy.fftpack
import glob
import os

# Ensure "signals" folder exists
if not os.path.exists("signals"):
    print("❌ Error: 'signals' folder not found! Run signal_generation.py first.")
    exit()

# Function to extract features
def extract_features(file_path, modulation_type):
    try:
        data = np.fromfile(file_path, dtype=np.complex64)[:500]  # Increased to 500 samples

        if len(data) == 0:
            return None  # Skip empty files

        # Feature 1: Signal Power (Energy)
        signal_power = np.mean(np.abs(data) ** 2)

        # Feature 2: FFT Mean (Frequency Component)
        fft_vals = np.abs(scipy.fftpack.fft(data))
        fft_mean = np.mean(fft_vals[:100])  # Consider first 100 FFT values

        # Feature 3: SNR (Signal-to-Noise Ratio)
        noise = data - np.mean(data)
        noise_power = np.mean(np.abs(noise) ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-6))  # Avoid division by zero

        return [signal_power, fft_mean, snr, modulation_type]

    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")
        return None

# Process all signal files
features = []
for mod in ["BPSK", "QPSK", "16-QAM"]:
    files = glob.glob(f"signals/{mod}_*.dat")

    if not files:
        print(f"⚠️ Warning: No files found for {mod}. Check signal generation!")
    
    for file_path in files:
        feature_vector = extract_features(file_path, mod)
        if feature_vector:
            features.append(feature_vector)

# Save extracted features to CSV
df = pd.DataFrame(features, columns=["Signal Power", "FFT Mean", "SNR", "Modulation"])
df.to_csv("features.csv", index=False)

print(f"✅ Feature extraction complete! Extracted {len(df)} feature sets. Data saved to `features.csv`.")
