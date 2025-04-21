import numpy as np
import pandas as pd
from gnuradio import gr, blocks, digital, analog
import joblib
import os
import time
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import io

# Define modulation types and parameters
modulation_types = ["BPSK", "QPSK", "16-QAM"]
samp_rate = 32000  # Samples per second
samples_per_chunk = int(samp_rate * 15)  # 480,000 samples per 15-second chunk
feature_columns = ["mean", "variance", "skewness", "kurtosis", "fft_mean", "spectral_entropy", 
                   "signal_power", "energy", "snr_db"]

# Output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
report_dir = os.path.join(output_dir, "reports")
os.makedirs(report_dir, exist_ok=True)

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
    snr = 10 * np.log10(signal_power / 0.01)
    return [mean, variance, skewness, kurtosis, fft_mean, spectral_entropy, signal_power, energy, snr]

# Dynamic signal generator
class DynamicSignalGenerator(gr.top_block):
    def __init__(self):
        gr.top_block.__init__(self, "Dynamic Signal Generator")
        self.samp_rate = samp_rate
        self.rng = np.random.RandomState(42)
        self.current_mod = self.rng.choice(modulation_types)
        self.current_snr_db = self.rng.uniform(0, 40)
        self.update_source()
        self.add = blocks.add_vcc(1)
        self.head = blocks.head(gr.sizeof_gr_complex * 1, samples_per_chunk)
        self.sink = blocks.vector_sink_c()
        self.connect(self.src, self.mod, (self.add, 0))
        self.connect(self.noise, (self.add, 1))
        self.connect(self.add, self.head, self.sink)

    def update_source(self):
        expected_bits = samples_per_chunk // 4
        if self.current_mod == "BPSK":
            self.original_bits = self.rng.randint(0, 2, expected_bits)
            self.src = blocks.vector_source_b(self.original_bits.tolist(), True)
            self.mod = digital.psk.psk_mod(constellation_points=2, samples_per_symbol=4)
            self.bits_per_symbol = 1
        elif self.current_mod == "QPSK":
            self.original_bits = self.rng.randint(0, 4, expected_bits)
            self.src = blocks.vector_source_b(self.original_bits.tolist(), True)
            self.mod = digital.psk.psk_mod(constellation_points=4, samples_per_symbol=4)
            self.bits_per_symbol = 2
        elif self.current_mod == "16-QAM":
            self.original_bits = self.rng.randint(0, 16, expected_bits)
            self.src = blocks.vector_source_b(self.original_bits.tolist(), True)
            constellation = digital.constellation_16qam().base()
            self.mod = digital.generic_mod(constellation=constellation, samples_per_symbol=4)
            self.bits_per_symbol = 4
        signal_power = 1.0
        snr_linear = 10 ** (self.current_snr_db / 20)
        noise_std = signal_power / snr_linear
        self.noise = analog.noise_source_c(analog.GR_GAUSSIAN, noise_std, 0)

    def sample_signal(self):
        self.current_mod = self.rng.choice(modulation_types)
        self.current_snr_db = self.rng.uniform(0, 40)
        self.update_source()
        self.disconnect_all()
        self.add = blocks.add_vcc(1)
        self.head = blocks.head(gr.sizeof_gr_complex * 1, samples_per_chunk)
        self.sink = blocks.vector_sink_c()
        self.connect(self.src, self.mod, (self.add, 0))
        self.connect(self.noise, (self.add, 1))
        self.connect(self.add, self.head, self.sink)
        self.start()
        self.wait()
        signal = np.array(self.sink.data())
        return signal, self.original_bits, self.bits_per_symbol, self.current_mod, self.current_snr_db

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

    def test(self, expected_length):
        self.start()
        self.wait()
        self.stop()
        demodulated_bits = np.array(self.sink.data())
        if len(demodulated_bits) > expected_length:
            demodulated_bits = demodulated_bits[:expected_length]
        elif len(demodulated_bits) < expected_length:
            demodulated_bits = np.pad(demodulated_bits, (0, expected_length - len(demodulated_bits)), mode='constant')
        return demodulated_bits, self.bits_per_symbol

# Load trained models
models = {
    "RandomForest": joblib.load(os.path.join(output_dir, "modulation_predictor_RandomForest.pkl")),
    "GradientBoosting": joblib.load(os.path.join(output_dir, "modulation_predictor_GradientBoosting.pkl")),
    "SVM": joblib.load(os.path.join(output_dir, "modulation_predictor_SVM.pkl"))
}

# Visualization and reporting class
class RealTimeDashboard:
    def __init__(self):
        self.times = []
        self.bers = []
        self.throughputs = []
        self.selected_mods = []
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(10, 8))
        self.fig.suptitle("Real-Time Signal Processing Dashboard")
        plt.ion()  # Interactive mode

    def update_plot(self, frame):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        self.ax1.plot(self.times, self.bers, 'b.-', label='BER')
        self.ax1.set_ylabel('BER')
        self.ax1.set_title('Bit Error Rate Over Time')
        self.ax1.legend()
        self.ax1.grid(True)

        self.ax2.plot(self.times, self.throughputs, 'g.-', label='Throughput (bps)')
        self.ax2.set_ylabel('Throughput (bps)')
        self.ax2.set_title('Throughput Over Time')
        self.ax2.legend()
        self.ax2.grid(True)

        self.ax3.plot(self.times, self.selected_mods, 'r.-', label='Selected Modulation')
        self.ax3.set_ylabel('Modulation Index')
        self.ax3.set_title('Selected Modulation Over Time')
        self.ax3.set_yticks(range(len(modulation_types)))
        self.ax3.set_yticklabels(modulation_types)
        self.ax3.legend()
        self.ax3.grid(True)

        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_xlabel('Time (s)')
        plt.tight_layout()

    def generate_pdf_report(self, timestamp, true_mod, snr_db, predictions, selected_mod, selected_ber, selected_throughput, perf_results):
        pdf_file = os.path.join(report_dir, f"report_{timestamp}.pdf")
        doc = SimpleDocTemplate(pdf_file, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph(f"Signal Classification Report - {time.ctime(timestamp)}", styles['Title']))
        story.append(Spacer(1, 12))

        # Signal Metadata
        story.append(Paragraph(f"<b>Signal Metadata:</b><br/>True Modulation: {true_mod}<br/>SNR: {snr_db:.2f} dB", styles['BodyText']))
        story.append(Spacer(1, 12))

        # Model Predictions
        pred_text = "<b>Model Predictions:</b><br/>" + "<br/>".join([f"{name}: {pred}" for name, pred in predictions.items()])
        pred_text += f"<br/>Selected Modulation: {selected_mod}"
        story.append(Paragraph(pred_text, styles['BodyText']))
        story.append(Spacer(1, 12))

        # Performance Table
        table_data = [["Modulation", "BER", "Throughput (bps)"]]
        for mod, perf in perf_results.items():
            table_data.append([mod, f"{perf['ber']:.4f}", f"{perf['throughput']:.0f}"])
        table_data.append(["Selected (" + selected_mod + ")", f"{selected_ber:.4f}", f"{selected_throughput:.0f}"])
        table = Table(table_data, colWidths=[100, 100, 100])
        table.setStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)])
        story.append(table)
        story.append(Spacer(1, 12))

        # BER Plot
        plt.figure(figsize=(5, 3))
        plt.plot([m for m in perf_results.keys()], [perf_results[m]['ber'] for m in perf_results.keys()], 'b.-', label='BER')
        plt.axhline(selected_ber, color='r', linestyle='--', label=f'Selected ({selected_mod})')
        plt.ylabel('BER')
        plt.title('BER Across Modulation Types')
        plt.legend()
        plt.grid(True)
        ber_buf = io.BytesIO()
        plt.savefig(ber_buf, format='png', bbox_inches='tight')
        ber_buf.seek(0)
        story.append(Image(ber_buf, 5*72, 3*72))
        plt.close()

        # Throughput Plot
        plt.figure(figsize=(5, 3))
        plt.plot([m for m in perf_results.keys()], [perf_results[m]['throughput'] for m in perf_results.keys()], 'g.-', label='Throughput')
        plt.axhline(selected_throughput, color='r', linestyle='--', label=f'Selected ({selected_mod})')
        plt.ylabel('Throughput (bps)')
        plt.title('Throughput Across Modulation Types')
        plt.legend()
        plt.grid(True)
        throughput_buf = io.BytesIO()
        plt.savefig(throughput_buf, format='png', bbox_inches='tight')
        throughput_buf.seek(0)
        story.append(Image(throughput_buf, 5*72, 3*72))
        plt.close()

        doc.build(story)

# Main execution
if __name__ == "__main__":
    print("Starting dynamic signal generation with visualization and reporting...")
    signal_gen = DynamicSignalGenerator()
    dashboard = RealTimeDashboard()
    
    try:
        iteration = 0
        while True:
            start_time = time.time()
            print(f"\n[{time.strftime('%H:%M:%S')}] Sampling 15-second signal...")
            signal, original_bits, true_bits_per_symbol, true_mod, snr_db = signal_gen.sample_signal()
            print(f"True Modulation: {true_mod}, SNR: {snr_db:.2f} dB")
            
            # Extract features
            features = extract_features(signal[:500])
            feature_df = pd.DataFrame([features], columns=feature_columns)

            # Predict modulation
            predictions = {name: model.predict(feature_df)[0] for name, model in models.items()}
            unique, counts = np.unique(list(predictions.values()), return_counts=True)
            selected_mod = unique[np.argmax(counts)]
            print("Model Predictions:")
            for name, pred in predictions.items():
                print(f"{name}: {pred}")
            print(f"Selected Modulation (Majority Vote): {selected_mod}")

            # Apply and demodulate
            tb_demod = DemodulationTester(selected_mod, signal)
            demodulated_bits, selected_bits_per_symbol = tb_demod.test(len(original_bits))
            selected_ber = np.sum(original_bits != demodulated_bits) / len(original_bits)
            selected_throughput = selected_bits_per_symbol * samp_rate / 4
            print(f"\nPerformance of Selected Modulation ({selected_mod}):")
            print(f"BER = {selected_ber:.2f}, Throughput = {selected_throughput:.0f} bps")

            # Verify against other modulation types
            perf_results = {}
            print("Verification Against Other Modulation Types:")
            for mod_type in modulation_types:
                tb_demod = DemodulationTester(mod_type, signal)
                demodulated_bits, bits_per_symbol = tb_demod.test(len(original_bits))
                ber = np.sum(original_bits != demodulated_bits) / len(original_bits)
                throughput = bits_per_symbol * samp_rate / 4
                perf_results[mod_type] = {"ber": ber, "throughput": throughput}
                print(f"{mod_type}: BER = {ber:.2f}, Throughput = {throughput:.0f} bps")

            # Update dashboard
            dashboard.times.append(iteration * 15)
            dashboard.bers.append(selected_ber)
            dashboard.throughputs.append(selected_throughput)
            dashboard.selected_mods.append(modulation_types.index(selected_mod))
            dashboard.update_plot(iteration)
            plt.pause(0.1)

            # Generate PDF report
            timestamp = int(time.time())
            dashboard.generate_pdf_report(timestamp, true_mod, snr_db, predictions, selected_mod, selected_ber, selected_throughput, perf_results)
            print(f"PDF Report Generated: {report_dir}/report_{timestamp}.pdf")

            # Wait for next 15-second cycle
            elapsed = time.time() - start_time
            if elapsed < 15:
                time.sleep(15 - elapsed)
            iteration += 1

    except KeyboardInterrupt:
        print("\nStopped by user.")
        signal_gen.stop()
        plt.close()