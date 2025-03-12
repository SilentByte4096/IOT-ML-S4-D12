import numpy as np
import os

# Ensure signal directory exists
signal_dir = "signals"
os.makedirs(signal_dir, exist_ok=True)

modulation_types = ["BPSK", "QPSK", "16-QAM"]
num_samples = 25
num_symbols = 1000  # Increase this if needed

for mod in modulation_types:
    for i in range(num_samples):
        np.random.seed()  # Remove fixed seed to introduce randomness

        # Generate random bits
        bits = np.random.randint(0, 2, num_symbols) if mod in ["BPSK", "QPSK"] else np.random.randint(0, 16, num_symbols)

        # Modulate signal
        if mod == "BPSK":
            signal = 2 * bits - 1  # Convert 0/1 to -1/+1
        elif mod == "QPSK":
            symbol_map = {0: 1+1j, 1: -1+1j, 2: -1-1j, 3: 1-1j}
            symbols = np.random.randint(0, 4, num_symbols)
            signal = np.array([symbol_map[s] for s in symbols])
        elif mod == "16-QAM":
            I = 2 * (bits % 4) - 3
            Q = 2 * (bits // 4) - 3
            signal = I + 1j * Q

        # Add noise
        noise = np.random.normal(0, 0.1, signal.shape) + 1j * np.random.normal(0, 0.1, signal.shape)
        noisy_signal = signal + noise

        # Save as binary file
        file_path = f"{signal_dir}/{mod}_{i}.dat"
        noisy_signal.astype(np.complex64).tofile(file_path)

print("✅ Signal generation complete! Now rerun feature extraction.")
