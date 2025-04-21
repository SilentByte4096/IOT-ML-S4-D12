import numpy as np
from gnuradio import gr, analog, blocks, digital
import os
import time
import pandas as pd
from tqdm import tqdm

# Directories
signal_dir = "signals"
metadata_dir = "metadata"
os.makedirs(signal_dir, exist_ok=True)
os.makedirs(metadata_dir, exist_ok=True)

# Parameters
modulation_types = ["BPSK", "QPSK", "16-QAM"]
num_samples_per_type = 500  
num_symbols = 500
samp_rate = 32000

# Signal generator class
class SignalGenerator(gr.top_block):
    def __init__(self, mod_type, snr_db, file_path, bits_file):
        gr.top_block.__init__(self, "Signal Generator")
        self.samp_rate = samp_rate
        
        # Bit source and modulation
        if mod_type == "BPSK":
            self.original_bits = np.random.randint(0, 2, num_symbols)
            self.src = blocks.vector_source_b(self.original_bits.tolist(), False)
            self.mod = digital.psk.psk_mod(constellation_points=2, samples_per_symbol=4)
        elif mod_type == "QPSK":
            self.original_bits = np.random.randint(0, 4, num_symbols)
            self.src = blocks.vector_source_b(self.original_bits.tolist(), False)
            self.mod = digital.psk.psk_mod(constellation_points=4, samples_per_symbol=4)
        elif mod_type == "16-QAM":
            self.original_bits = np.random.randint(0, 16, num_symbols)
            self.src = blocks.vector_source_b(self.original_bits.tolist(), False)
            constellation = digital.constellation_16qam().base()
            self.mod = digital.generic_mod(constellation=constellation, samples_per_symbol=4)
        
        # Noise based on SNR
        signal_power = 1.0
        snr_linear = 10 ** (snr_db / 20) 
        noise_std = signal_power / snr_linear
        self.noise = analog.noise_source_c(analog.GR_GAUSSIAN, noise_std, 0)
        
        # Flowgraph
        self.add = blocks.add_vcc(1)
        self.head = blocks.head(gr.sizeof_gr_complex * 1, num_symbols * 4)
        self.sink = blocks.file_sink(gr.sizeof_gr_complex * 1, file_path)
        
        self.connect(self.src, self.mod, (self.add, 0))
        self.connect(self.noise, (self.add, 1))
        self.connect(self.add, self.head, self.sink)
        
        # Save original bits
        np.save(bits_file, self.original_bits)

    def run_and_stop(self):
        self.start()
        self.wait()
        self.stop()

# Generate and save signal
def generate_and_save_signal(mod_type, i, snr_db):
    timestamp = int(time.time() * 1000)
    file_path = os.path.join(signal_dir, f"{mod_type}_{i}_{timestamp}.dat")
    bits_file = os.path.join(metadata_dir, f"{mod_type}_{i}_{timestamp}_bits.npy")
    meta_path = os.path.join(metadata_dir, f"{mod_type}_{i}_{timestamp}.csv")
    
    tb = SignalGenerator(mod_type, snr_db, file_path, bits_file)
    tb.run_and_stop()
    
    metadata = {
        "signal_type": mod_type,
        "snr_db": snr_db,
        "samples": num_symbols,
        "timestamp": timestamp,
        "file_path": file_path,
        "bits_file": bits_file
    }
    pd.DataFrame([metadata]).to_csv(meta_path, index=False)

# Main execution
if __name__ == "__main__":
    rng = np.random.default_rng()
    total_signals = num_samples_per_type * len(modulation_types)
    
    with tqdm(total=total_signals, desc="Generating Signals") as pbar:
        for mod_type in modulation_types:
            for i in range(num_samples_per_type):
                snr_db = rng.uniform(0, 40)
                generate_and_save_signal(mod_type, i, snr_db)
                pbar.update(1)
    
    print(f"Signal generation complete! {total_signals} signals generated.")