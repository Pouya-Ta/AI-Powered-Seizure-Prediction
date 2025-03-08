import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch
import re
import warnings

warnings.filterwarnings("ignore", message="Scaling factor is not defined")

# Define paths
data_folder = r"Your Dir"  # Change to your raw data folder
preprocessed_folder = r"Your Dir"  # Preprocessed data storage
plots_folder = r"Your Dir"
seizure_summary_csv_path = r"Your Dir"  # Global CSV summary file

os.makedirs(preprocessed_folder, exist_ok=True)
os.makedirs(plots_folder, exist_ok=True)


def band_filter(data, low_freq, high_freq, sfreq):
    """Applies a bandpass filter."""
    nyquist = sfreq / 2
    low, high = low_freq / nyquist, high_freq / nyquist
    b, a = butter(N=4, Wn=[low, high], btype="bandpass")
    return filtfilt(b, a, data, axis=1)


def notch_filter(data, sfreq):
    """Applies a notch filter at 60 Hz."""
    b, a = iirnotch(60, 30, sfreq)
    return filtfilt(b, a, data, axis=1)


def preprocess_eeg(raw_data, fs):
    """Applies bandpass and notch filters."""
    filtered_data = band_filter(raw_data, 0.5, 80, fs)
    centered_data = filtered_data - np.mean(filtered_data, axis=1, keepdims=True)
    cleaned_data = notch_filter(centered_data, fs)
    return cleaned_data


def create_windows(data, fs, window_size):
    """Splits data into fixed-size windows."""
    step = int(window_size * fs)
    return [data[:, i : i + step] for i in range(0, data.shape[1] - step + 1, step)]


def plot_window(window, fs, ch_names, save_path):
    """Plots a given EEG window and saves it."""
    time = np.linspace(0, window.shape[1] / fs, window.shape[1])
    plt.figure(figsize=(10, 6))
    for i, channel in enumerate(window):
        plt.plot(time, channel, label=ch_names[i])
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.title(os.path.basename(save_path))
    plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1.0), fontsize="small", frameon=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def read_summary_csv(csv_path):
    """
    Reads CSV summary file and extracts seizure information.
    
    Expected CSV format:
      File Name,Seizure Start Time,Seizure End Time
    """
    df = pd.read_csv(csv_path)
    seizure_info = {}
    for _, row in df.iterrows():
        file_name = row["File_name"]
        start_time = int(row["Seizure_start"])
        end_time = int(row["Seizure_stop"])
        if file_name in seizure_info:
            seizure_info[file_name].append((start_time, end_time))
        else:
            seizure_info[file_name] = [(start_time, end_time)]
    return seizure_info


def segment_eeg(file_name, preprocessed_data, fs, patient_folder, ch_names, seizure_info):
    """Segments EEG into ictal, preictal, and interictal windows."""
    if file_name not in seizure_info or not seizure_info[file_name]:
        print(f"Info: {file_name} has no seizures, skipping segmentation.")
        return

    patient_preprocessed_folder = os.path.join(preprocessed_folder, patient_folder)
    os.makedirs(patient_preprocessed_folder, exist_ok=True)

    window_size = 2  # seconds
    for seizure_start, seizure_end in seizure_info[file_name]:
        ictal_data = preprocessed_data[:, seizure_start * fs : seizure_end * fs]
        ictal_windows = create_windows(ictal_data, fs, window_size)

        preictal_start = max(0, seizure_start - len(ictal_windows) * window_size)
        preictal_data = preprocessed_data[:, preictal_start * fs : seizure_start * fs]
        preictal_windows = create_windows(preictal_data, fs, window_size)

        interictal_start = max(0, preictal_start - len(ictal_windows) * window_size * 30)
        interictal_data = preprocessed_data[:, interictal_start * fs : preictal_start * fs]
        interictal_windows = create_windows(interictal_data, fs, window_size)

        # Save and plot each segment
        for i, window in enumerate(ictal_windows):
            save_path = os.path.join(patient_preprocessed_folder, f"{file_name}_ictal_{i}.npy")
            np.save(save_path, window)
            plot_window(window, fs, ch_names, save_path.replace(".npy", "_plot.png"))

        for i, window in enumerate(preictal_windows):
            save_path = os.path.join(patient_preprocessed_folder, f"{file_name}_preictal_{i}.npy")
            np.save(save_path, window)
            plot_window(window, fs, ch_names, save_path.replace(".npy", "_plot.png"))

        for i, window in enumerate(interictal_windows):
            save_path = os.path.join(patient_preprocessed_folder, f"{file_name}_interictal_{i}.npy")
            np.save(save_path, window)
            plot_window(window, fs, ch_names, save_path.replace(".npy", "_plot.png"))


# Load the global seizure information from the CSV file
global_seizure_info = read_summary_csv(seizure_summary_csv_path)

# Sorting folders numerically
def extract_number(folder_name):
    match = re.search(r"\d+", folder_name)  # Extract numeric part
    return int(match.group()) if match else float("inf")

sorted_folders = sorted(os.listdir(data_folder), key=extract_number)

for patient_folder in sorted_folders:
    print(f"Processing folder: {patient_folder}")
    patient_path = os.path.join(data_folder, patient_folder)

    if not os.path.isdir(patient_path):
        print(f"Skipping {patient_folder}: Not a directory")
        continue

    # Process each .edf file in the patient folder
    for filename in os.listdir(patient_path):
        if not filename.endswith(".edf"):
            continue

        file_path = os.path.join(patient_path, filename)

        # Skip files that have no seizure info in the global CSV
        if filename not in global_seizure_info or not global_seizure_info[filename]:
            print(f"Skipping {filename}: No seizures")
            continue

        try:
            raw = mne.io.read_raw_edf(file_path, preload=True)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        # Remove invalid or duplicate channel names
        raw.drop_channels([ch for ch in raw.info["ch_names"] if ch == "-"])

        eeg_data = raw.get_data()
        fs = int(raw.info["sfreq"])
        ch_names = raw.ch_names

        preprocessed_data = preprocess_eeg(eeg_data, fs)

        segment_eeg(filename, preprocessed_data, fs, patient_folder, ch_names, global_seizure_info)

print("Preprocessing, segmentation, and plotting complete!")
