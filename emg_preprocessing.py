# emg_preprocessing.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt



def remove_mean(signal):
    return signal - np.mean(signal)



def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


def moving_rms(signal, window_size=100):
    squared_signal = signal ** 2
    window = np.ones(window_size) / window_size
    squared_mean = np.convolve(squared_signal, window, mode='same')
    return np.sqrt(squared_mean)



def process_and_plot_emg(csv_path, title, window_size = 20):

    """
    1. Load data (assuming single column with no header).
    2. Remove offset.
    3. Rectify.
    4. RMS envelope.
    5. Plot subplots in one figure.
    """
    fs = 1000
    # 1) Load data
    df = pd.read_csv(csv_path, header=None, names=["raw"])
    # Remove any "A0:" prefix
    df["raw"] = df["raw"].astype(str).str.replace("A0:", "", regex=False)
    signal_raw = df["raw"].astype(float).values

    # 2) Remove offset
    signal_no_offset = remove_mean(signal_raw)

    # 3) Filter signal
    signal_filtered = butter_bandpass_filter(signal_no_offset, 20, 450,  1000)


    # 4) Rectify
    signal_rectified = np.abs(signal_filtered)

    # 5) RMS envelope
    rms_env = moving_rms(signal_rectified, window_size = window_size)

    # 6) Plot
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(signal_no_offset, label="Offset Removed ")
    axs[0].set_title(f"{title} - Offset Removed Signal")
    axs[0].set_ylabel("Amplitude")

    axs[1].plot(signal_rectified, label="Rectified", color="green")
    axs[1].set_title(f"{title} - Rectified Signal")
    axs[1].set_ylabel("Amplitude")

    axs[2].plot(rms_env, label="RMS Envelope", color="red")
    axs[2].set_title(f"{title} - RMS Envelope")
    axs[2].set_xlabel("Sample index")
    axs[2].set_ylabel("Amplitude ")

    plt.tight_layout()
    plt.show()


def filtered_signal(csv_path, window_size):
    df = pd.read_csv(csv_path, header=None, names=["raw"])
    # Remove any "A0:" prefix
    df["raw"] = df["raw"].astype(str).str.replace("A0:", "", regex=False)
    signal_raw = df["raw"].astype(float).values

    # 2) Remove offset
    signal_no_offset = remove_mean(signal_raw)

    # 3) Filter signal
    signal_filtered = butter_bandpass_filter(signal_no_offset, 20, 450, 1000)

    # 4) Rectify
    signal_rectified = np.abs(signal_filtered)

    # 5) RMS envelope
    rms_env = moving_rms(signal_rectified,window_size)

    return rms_env

def analyze_three_bursts(signal,
                         start_1, end_1,
                         start_2, end_2,
                         start_3, end_3,
                         title="Filtered Signal (3 Bursts)"):
    # --- Compute the means for each burst ---
    burst1 = signal[start_1:end_1]
    burst2 = signal[start_2:end_2]
    burst3 = signal[start_3:end_3]

    mean_1 = np.mean(burst1)
    mean_2 = np.mean(burst2)
    mean_3 = np.mean(burst3)

    # overall mean of all 3 bursts
    # (you could do np.mean of the concatenation or just the average of the 3 means)
    overall_mean = np.mean([mean_1, mean_2, mean_3])

    # --- Plot ---
    plt.figure(figsize=(10, 4))
    plt.plot(signal, label="Filtered EMG")

    # Draw vertical lines at each boundary
    # (use different colors so you can distinguish the three bursts)
    plt.axvline(x=start_1, color='r', linestyle='--', label="Burst 1 Start")
    plt.axvline(x=end_1, color='r', linestyle='-', label="Burst 1 End")

    plt.axvline(x=start_2, color='g', linestyle='--', label="Burst 2 Start")
    plt.axvline(x=end_2, color='g', linestyle='-', label="Burst 2 End")

    plt.axvline(x=start_3, color='b', linestyle='--', label="Burst 3 Start")
    plt.axvline(x=end_3, color='b', linestyle='-', label="Burst 3 End")

    plt.title(title)
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

    # Print or return the means
    print(f"Mean burst1: {mean_1:.3f}")
    print(f"Mean burst2: {mean_2:.3f}")
    print(f"Mean burst3: {mean_3:.3f}")
    print(f"Overall mean: {overall_mean:.3f}")

    return mean_1, mean_2, mean_3, overall_mean