import os
import numpy as np
import pandas as pd
import pywt
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import spectrogram, welch

# EEG frequency bands
bands = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 100),
}

# Hjorth Parameters
def hjorth_parameters(signal):
    activity = np.var(signal)
    mobility = np.sqrt(np.var(np.diff(signal)) / activity)
    complexity = np.sqrt(np.var(np.diff(np.diff(signal))) / np.var(np.diff(signal)))
    return activity, mobility, complexity

# Zero Crossing Rate
def zero_crossing_rate(signal):
    return np.sum(np.diff(np.sign(signal)) != 0) / len(signal)

# Frequency Domain Features (Welch PSD)
def compute_frequency_features(signal, fs):
    freqs, psd = welch(signal, fs=fs, nperseg=1024)
    mean_freq = np.sum(freqs * psd) / np.sum(psd)
    median_freq = freqs[np.where(np.cumsum(psd) >= np.sum(psd) / 2)[0][0]]
    peak_freq = freqs[np.argmax(psd)]
    psd_norm = psd / np.sum(psd)
    spec_entropy = entropy(psd_norm)
    band_powers = {
        band: np.sum(psd[(freqs >= low) & (freqs <= high)]) / np.sum(psd)
        for band, (low, high) in bands.items()
    }
    return mean_freq, median_freq, peak_freq, spec_entropy, band_powers

# Time-Frequency Features (STFT)
def compute_stft_features(signal, fs):
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=256)
    Sxx_norm = Sxx / np.sum(Sxx)
    spec_entropy = np.mean(entropy(Sxx_norm, axis=0))
    mean_freq = np.mean(np.sum(f[:, None] * Sxx, axis=0) / np.sum(Sxx, axis=0))
    median_freq = np.mean(
        f[np.argmax(np.cumsum(Sxx, axis=0) >= np.sum(Sxx, axis=0) / 2, axis=0)]
    )
    peak_freq = np.mean(f[np.argmax(Sxx, axis=0)])
    return mean_freq, median_freq, peak_freq, spec_entropy

def compute_wavelet_energy(signal, wavelet="db4", level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return [np.sum(np.square(c)) for c in coeffs[1:]]

# Extract channel names from the summary file
def extract_channel_names(summary_path):
    """Extracts channel names from the summary text file."""
    channel_names = []
    with open(summary_path, "r") as file:
        lines = file.readlines()
        capture = False
        for line in lines:
            if line.startswith("Channels in EDF Files"):
                capture = True
            if capture and line.startswith("Channel"):
                parts = line.split(":")
                if len(parts) > 1:
                    channel_name = parts[1].strip()
                    channel_names.append(channel_name)
            if capture and line.startswith("File Name:"):
                # Stop capturing once the next file is reached
                break
    return channel_names

# Define paths
preprocessed_folder = r"Your Dir"
output_folder = r"Your Dir"

# For each patient folder
for patient in os.listdir(preprocessed_folder):
    patient_path = os.path.join(preprocessed_folder, patient)
    if os.path.isdir(patient_path):
        save_path = os.path.join(output_folder, patient)
        os.makedirs(save_path, exist_ok=True)

        # Gather list of .npy files and determine max channels across files
        npy_files = [f for f in os.listdir(patient_path) if f.endswith(".npy")]
        if not npy_files:
            print(f"No .npy files found in {patient_path}.")
            continue
        max_channels = 0
        for file in npy_files:
            data = np.load(os.path.join(patient_path, file))
            if data.shape[0] > max_channels:
                max_channels = data.shape[0]

        # Try to get channel names from the summary file
        summary_files = [f for f in os.listdir(patient_path) if f.endswith(".txt")]
        channel_names = []
        if summary_files:
            summary_path = os.path.join(patient_path, summary_files[0])
            channel_names = extract_channel_names(summary_path)

        # If channel_names is empty or its length does not match max_channels, adjust it.
        if channel_names:
            if len(channel_names) < max_channels:
                for i in range(len(channel_names), max_channels):
                    channel_names.append(f"Ch{i+1}")
            elif len(channel_names) > max_channels:
                channel_names = channel_names[:max_channels]
        else:
            channel_names = [f"Ch{i+1}" for i in range(max_channels)]

        # List to collect all rows (one per window) for the patient
        patient_features = []

        # Process each .npy file (each representing one window)
        for file in npy_files:
            file_path = os.path.join(patient_path, file)
            window_data = np.load(file_path)  # Expected shape: (channels, samples)
            fs = 256  # Adjust sample rate if needed

            # Start the row with the window (file) name
            row_features = [file]
            n_channels = window_data.shape[0]

            # Compute features for each channel in this file
            for i in range(n_channels):
                signal = window_data[i]
                # Time-domain features
                mean_val = np.mean(signal)
                variance = np.var(signal)
                std_dev = np.std(signal)
                rms = np.sqrt(np.mean(signal**2))
                skewness = skew(signal)
                kurt = kurtosis(signal)
                peak_to_peak = np.ptp(signal)
                zcr = zero_crossing_rate(signal)
                activity, mobility, complexity = hjorth_parameters(signal)

                # Frequency-domain features
                mean_freq, median_freq, peak_freq, spec_entropy, band_powers = compute_frequency_features(signal, fs)

                # Time-frequency features (STFT)
                mean_tf, median_tf, peak_tf, spec_entropy_tf = compute_stft_features(signal, fs)

                # Wavelet energy features (4 levels)
                wavelet_energy = compute_wavelet_energy(signal)

                # Combine features for this channel (28 features)
                channel_features = [
                    mean_val,
                    variance,
                    std_dev,
                    rms,
                    skewness,
                    kurt,
                    peak_to_peak,
                    zcr,
                    activity,
                    mobility,
                    complexity,
                    mean_freq,
                    median_freq,
                    peak_freq,
                    spec_entropy,
                ] + list(band_powers.values()) + [mean_tf, median_tf, peak_tf, spec_entropy_tf] + wavelet_energy

                row_features.extend(channel_features)

            # If this file has fewer channels than max_channels, pad with NaNs for the missing channels
            if n_channels < max_channels:
                missing_channels = max_channels - n_channels
                row_features.extend([np.nan] * (missing_channels * 28))

            # Now each row should have: 1 + (max_channels * 28) features
            patient_features.append(row_features)

        # If no features were extracted, warn and skip CSV creation.
        if not patient_features:
            print(f"No feature data extracted for patient {patient}.")
            continue

        # Dynamically generate column names based on max_channels.
        columns = ["Window_Name"]
        time_features = [
            "Mean",
            "Variance",
            "STD",
            "RMS",
            "Skewness",
            "Kurtosis",
            "Peak-to-Peak",
            "ZCR",
            "Activity",
            "Mobility",
            "Complexity",
        ]
        freq_features = ["Mean Frequency", "Median Frequency", "Peak Frequency", "Spectral Entropy"]
        band_features = list(bands.keys())
        tf_features = ["Mean Frequency (T-F)", "Median Frequency (T-F)", "Peak Frequency (T-F)", "Spectral Entropy (T-F)"]
        wavelet_features = [f"Wavelet Energy L{lvl}" for lvl in range(1, 5)]

        # Each channel contributes 28 features: 11 (time) + 4 (frequency) + 5 (bands) + 4 (T-F) + 4 (wavelet)
        for ch in channel_names:
            for feat in time_features:
                columns.append(f"{feat}_channel_{ch}")
            for feat in freq_features:
                columns.append(f"{feat}_channel_{ch}")
            for feat in band_features:
                columns.append(f"{feat}_channel_{ch}")
            for feat in tf_features:
                columns.append(f"{feat}_channel_{ch}")
            for feat in wavelet_features:
                columns.append(f"{feat}_channel_{ch}")

        expected_total_columns = 1 + max_channels * 28
        if len(columns) != expected_total_columns:
            print("Warning: Column count does not match expected feature count.")

        # Create DataFrame from the collected rows and save as CSV.
        final_df = pd.DataFrame(patient_features, columns=columns)
        output_file = os.path.join(save_path, f"{patient}_features.csv")
        final_df.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")
