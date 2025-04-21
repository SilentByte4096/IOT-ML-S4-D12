import numpy as np
import pandas as pd
from gnuradio import gr, blocks, digital, analog
import joblib
import os
import time
from scipy import stats

# Define modulation types and parameters
modulation_types = ["BPSK", "QPSK", "16-QAM"]
samp_rate = 32000  # Samples per second
duration = 15      # 15 seconds
num_samples = int(samp_rate * duration)  # 480,000 samples
feature_columns = ["mean", "variance", "skewness", "kurtosis", "fft_mean", "spectral_entropy", 
                   "signal_power", "energy", "snr_db"]

# Output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
fallback_log_file = os.path.join(output_dir, "fallback_log.csv")

# Feature extraction function
def extract_features(signal):
    real_signal = np.real(signal)
    mean = np.mean(real_signal)
    variance = np.var(real_signal)
    skewness = stats.skew(real_signal)
    kurtosis = stats.kurtosis(real_signal)
    fft = np.abs(np.fft.fft(signal))
    fft_mean = np.mean(fft)
    spectral_entropy = stats.entropy(np.abs(fft) / np.sum(np.abs(fft)))
    signal_power = np.mean(np.abs(signal)**2)
    energy = np.sum(np.abs(signal)**2)
    snr = 10 * np.log10(signal_power / 0.01)  # Estimated SNR
    return [mean, variance, skewness, kurtosis, fft_mean, spectral_entropy, signal_power, energy, snr]

# Signal generator class (simulated real-time input)
class SignalGenerator(gr.top_block):
    def __init__(self, mod_type, snr_db):
        gr.top_block.__init__(self, "Signal Generator")
        self.samp_rate = samp_rate
        
        if mod_type == "BPSK":
            self.original_bits = np.random.randint(0, 2, num_samples // 4)
            self.src = blocks.vector_source_b(self.original_bits.tolist(), False)
            self.mod = digital.psk.psk_mod(constellation_points=2, samples_per_symbol=4)
            self.bits_per_symbol = 1
        elif mod_type == "QPSK":
            self.original_bits = np.random.randint(0, 4, num_samples // 4)
            self.src = blocks.vector_source_b(self.original_bits.tolist(), False)
            self.mod = digital.psk.psk_mod(constellation_points=4, samples_per_symbol=4)
            self.bits_per_symbol = 2
        elif mod_type == "16-QAM":
            self.original_bits = np.random.randint(0, 16, num_samples // 4)
            self.src = blocks.vector_source_b(self.original_bits.tolist(), False)
            constellation = digital.constellation_16qam().base()
            self.mod = digital.generic_mod(constellation=constellation, samples_per_symbol=4)
            self.bits_per_symbol = 4
        
        signal_power = 1.0
        snr_linear = 10 ** (snr_db / 20)
        noise_std = signal_power / snr_linear
        self.noise = analog.noise_source_c(analog.GR_GAUSSIAN, noise_std, 0)
        
        self.add = blocks.add_vcc(1)
        self.sink = blocks.vector_sink_c()
        self.connect(self.src, self.mod, (self.add, 0))
        self.connect(self.noise, (self.add, 1))
        self.connect(self.add, self.sink)

    def run_and_get_signal(self):
        self.start()
        self.wait()
        self.stop()
        return np.array(self.sink.data()), self.original_bits, self.bits_per_symbol

# Demodulation tester
class DemodulationTester(gr.top_block):
    def __init__(self, mod_type, signal):
        gr.top_block.__init__(self, "Demodulation Tester")
        self.src = blocks.vector_source_c(signal.tolist(), False)
        
        if mod_type == "BPSK":
            self.demod = digital.psk.psk_demod(constellation_points=2, samples_per_symbol=4)
            self.bits_per_symbol = 1
        elif mod_type == "QPSK":
            self.demod = digital.psk.psk_demod(constellation_points=4, samples_per_symbol=4)
            self.bits_per_symbol = 2
        elif mod_type == "16-QAM":
            constellation = digital.constellation_16qam().base()
            self.demod = digital.generic_demod(constellation=constellation, samples_per_symbol=4)
            self.bits_per_symbol = 4
        
        self.sink = blocks.vector_sink_b()
        self.connect(self.src, self.demod, self.sink)

    def test(self):
        self.start()
        self.wait()
        self.stop()
        demodulated_bits = np.array(self.sink.data())[:num_samples // 4]
        return demodulated_bits, self.bits_per_symbol

# Load trained models
models = {
    "RandomForest": joblib.load(os.path.join(output_dir, "modulation_predictor_RandomForest.pkl")),
    "GradientBoosting": joblib.load(os.path.join(output_dir, "modulation_predictor_GradientBoosting.pkl")),
    "SVM": joblib.load(os.path.join(output_dir, "modulation_predictor_SVM.pkl"))
}

# Main execution
if __name__ == "__main__":
    rng = np.random.RandomState(42)
    true_mod = rng.choice(modulation_types)
    snr_db = rng.uniform(0, 40)

    # Step 4: Adaptive Real-Time Processing
    print(f"Sampling 15-second real-time signal (simulated) with {true_mod} and SNR {snr_db:.2f} dB...")
    tb_gen = SignalGenerator(true_mod, snr_db)
    signal, original_bits, true_bits_per_symbol = tb_gen.run_and_get_signal()

    # Extract features from the sampled signal
    features = extract_features(signal[:500])  # Use first 500 samples for consistency
    feature_df = pd.DataFrame([features], columns=feature_columns)

    # Predict modulation using trained models
    predictions = {name: model.predict(feature_df)[0] for name, model in models.items()}
    unique, counts = np.unique(list(predictions.values()), return_counts=True)
    selected_mod = unique[np.argmax(counts)]
    print("\nModel Predictions:")
    for name, pred in predictions.items():
        print(f"{name}: {pred}")
    print(f"Selected Modulation (Majority Vote): {selected_mod}")

    # Apply selected modulation (simulate by demodulating with selected scheme)
    tb_demod = DemodulationTester(selected_mod, signal)
    demodulated_bits, selected_bits_per_symbol = tb_demod.test()

    # Step 5: Performance Verification
    selected_ber = np.sum(original_bits != demodulated_bits) / len(original_bits)
    selected_throughput = selected_bits_per_symbol * samp_rate / 4
    print(f"\nPerformance of Selected Modulation ({selected_mod}):")
    print(f"BER = {selected_ber:.2f}, Throughput = {selected_throughput:.0f} bps")

    # Compute BER and throughput for all schemes
    perf_results = {}
    for mod_type in modulation_types:
        tb_demod = DemodulationTester(mod_type, signal)
        demodulated_bits, bits_per_symbol = tb_demod.test()
        ber = np.sum(original_bits != demodulated_bits) / len(original_bits)
        throughput = bits_per_symbol * samp_rate / 4
        perf_results[mod_type] = {"ber": ber, "throughput": throughput}
        print(f"{mod_type}: BER = {ber:.2f}, Throughput = {throughput:.0f} bps")
