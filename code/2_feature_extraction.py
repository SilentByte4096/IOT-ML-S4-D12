import numpy as np
import pandas as pd
from scipy import stats
from gnuradio import gr, blocks, digital
import os
from tqdm import tqdm

# Directories
signal_dir = "signals"
metadata_dir = "metadata"
output_file = "features_with_modulation.csv"

# Modulation schemes
modulation_types = ["BPSK", "QPSK", "16-QAM"]
samp_rate = 32000
num_symbols = 500

# Extract features from demodulated signal
def extract_features(demodulated_bits):
    signal = demodulated_bits.astype(np.float32)  
    mean = np.mean(signal)
    variance = np.var(signal)
    skewness = stats.skew(signal)
    kurtosis = stats.kurtosis(signal)
    fft = np.abs(np.fft.fft(signal))
    fft_mean = np.mean(fft)
    spectral_entropy = stats.entropy(np.abs(fft) / np.sum(np.abs(fft)))
    signal_power = np.mean(signal**2)
    energy = np.sum(signal**2)
    return [mean, variance, skewness, kurtosis, fft_mean, spectral_entropy, signal_power, energy]

# Demodulation tester
class DemodulationTester(gr.top_block):
    def __init__(self, mod_type, signal):
        gr.top_block.__init__(self, "Demodulation Tester")
        self.src = blocks.vector_source_c(signal.tolist(), False)
        
        # Demodulation
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
        
        # Flowgraph
        self.head = blocks.head(gr.sizeof_gr_complex * 1, num_symbols * 4)
        self.sink = blocks.vector_sink_b()
        self.connect(self.src, self.head, self.demod, self.sink)

    def test(self):
        self.start()
        self.wait()
        self.stop()
        demodulated_bits = np.array(self.sink.data())
        if len(demodulated_bits) > num_symbols:
            demodulated_bits = demodulated_bits[:num_symbols]
        elif len(demodulated_bits) < num_symbols:
            demodulated_bits = np.pad(demodulated_bits, (0, num_symbols - len(demodulated_bits)), 'constant')
        return demodulated_bits, self.bits_per_symbol

# Process signal
def process_signal(signal, mod_type, original_bits, snr_db):
    tb = DemodulationTester(mod_type, signal)
    demodulated_bits, bits_per_symbol = tb.test()
    
    # BER
    error_count = np.sum(original_bits != demodulated_bits)
    ber = error_count / num_symbols
    
    # Throughput
    throughput = bits_per_symbol * samp_rate / 4  
    
    # Features from demodulated signal
    features = extract_features(demodulated_bits)
    
    return features, ber, throughput

# Main execution
if __name__ == "__main__":
    signal_files = [f for f in os.listdir(signal_dir) if f.endswith(".dat")]
    data = []
    
    for signal_file in tqdm(signal_files, desc="Extracting Features"):
        signal_path = os.path.join(signal_dir, signal_file)
        signal = np.fromfile(signal_path, dtype=np.complex64)[:num_symbols * 4]  
        
        meta_file = os.path.join(metadata_dir, signal_file.replace(".dat", ".csv"))
        metadata = pd.read_csv(meta_file).iloc[0]
        snr_db = metadata["snr_db"]
        mod_type = metadata["signal_type"]
        bits_file = metadata["bits_file"]
        original_bits = np.load(bits_file)
        
        features, ber, throughput = process_signal(signal, mod_type, original_bits, snr_db)
        
        data.append(features + [snr_db, mod_type, ber, throughput])
    
    columns = ["mean", "variance", "skewness", "kurtosis", "fft_mean", "spectral_entropy", 
               "signal_power", "energy", "snr_db", "modulation_type", "ber", "throughput"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_file, index=False)
    print(f"✔ Feature extraction complete! Saved to {output_file}")