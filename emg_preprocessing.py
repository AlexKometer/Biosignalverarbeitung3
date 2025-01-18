# emg_preprocessing.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import matplotlib.patches as mpatches
from scipy import fftpack


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

def process_and_plot_emg_weight(csv_path, title, window_size = 20, maximum_mvc = 20):

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
    signal_filtered = butter_bandpass_filter(signal_no_offset, 20, 450, 1000)

    # 4) Rectify
    signal_rectified = np.abs(signal_filtered)

    # 5) RMS envelope
    rms_env = moving_rms(signal_rectified, window_size=window_size)

    # Normalize signals to percentage of maximum MVC
    signal_no_offset_percentage = (signal_no_offset / maximum_mvc) * 100
    signal_rectified_percentage = (signal_rectified / maximum_mvc) * 100
    rms_env_percentage = (rms_env / maximum_mvc) * 100

    # 6) Plot RMS Envelope with shaded areas
    plt.figure(figsize=(10, 6))

    plt.plot(rms_env_percentage, label="RMS Envelope", color="red")
    plt.title(f"{title} - RMS Envelope")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude (%)")
    plt.ylim(0, 100)

    # Add shaded areas with legends
    green_patch = mpatches.Patch(color='green', alpha=0.5, label='2,5 KG')
    orange_patch = mpatches.Patch(color='orange', alpha=0.5, label='5 KG')
    blue_patch = mpatches.Patch(color='blue', alpha=0.5, label='10 KG')

    plt.axvspan(125, 230, color='green', alpha=0.5)  # First highlighted region
    plt.axvspan(360, 470, color='orange', alpha=0.5)  # Second highlighted region
    plt.axvspan(630, 740, color='blue', alpha=0.5)    # Third highlighted region

    # Adding the legend for the shaded areas
    plt.legend(handles=[green_patch, orange_patch, blue_patch, plt.Line2D([0], [0], color="red", lw=2, label="RMS Envelope")])

    plt.tight_layout()
    plt.ylim(0, maximum_mvc)
    plt.show()



# Funktion zur Berechnung der Leistung und Frequenzen
def get_power(data):
    sig_fft = fftpack.fft(data)

    # Leistung (sig_fft ist vom Typ complex)
    power = np.abs(sig_fft)

    # Die entsprechenden Frequenzen
    sample_freq1 = fftpack.fftfreq(data.size, d=1 / 1000)  # Annahme: Sampling-Rate = 1000 Hz
    frequencies = sample_freq1[sample_freq1 > 0]
    power = power[sample_freq1 > 0]
    return power, frequencies


# Butterworth-Filter mit einer Cutoff-Frequenz von 40 Hz
def butter_bandpass_filter(data, lowcut=0.1, highcut=40.0, fs=1000, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


# Funktion zur Erstellung von Plots für die 3 Datensätze und deren FFTs
def plot_fatigue_fft(fatigue_1, fatigue_2, fatigue_3):
    # Berechne die Länge jedes Datensatzes
    len_fatigue_1 = len(fatigue_1)
    len_fatigue_2 = len(fatigue_2)
    len_fatigue_3 = len(fatigue_3)

    # Teile die Daten in Start, Mitte und Ende
    data_sets = [fatigue_1, fatigue_2, fatigue_3]
    titles = ['Fatigue 1', 'Fatigue 2', 'Fatigue 3']
    timepoints = ['Start', 'Middle', 'End']

    # Erstelle die 9 Plots (3 Datensätze, 3 Zeitpunkte je Datensatz)
    fig, axs = plt.subplots(3, 3, figsize=(15, 12))

    for i, data in enumerate(data_sets):
        # Start, Mitte, Ende
        start = data[:len(data) // 3]
        middle = data[len(data) // 3:2 * len(data) // 3]
        end = data[2 * len(data) // 3:]

        # Berechne FFT und Leistung für die 3 Zeitpunkte
        start_power, start_freq = get_power(start)
        middle_power, middle_freq = get_power(middle)
        end_power, end_freq = get_power(end)

        # Filtere das Signal
        start_filtered = butter_bandpass_filter(start, lowcut=0.1, highcut=40.0)
        middle_filtered = butter_bandpass_filter(middle, lowcut=0.1, highcut=40.0)
        end_filtered = butter_bandpass_filter(end, lowcut=0.1, highcut=40.0)

        # Berechne die Leistung des gefilterten Signals
        start_filtered_power, _ = get_power(start_filtered)
        middle_filtered_power, _ = get_power(middle_filtered)
        end_filtered_power, _ = get_power(end_filtered)

        # Berechne die Mediane der Leistungsdaten
        start_median = np.median(start_power)
        middle_median = np.median(middle_power)
        end_median = np.median(end_power)

        start_filtered_median = np.median(start_filtered_power)
        middle_filtered_median = np.median(middle_filtered_power)
        end_filtered_median = np.median(end_filtered_power)

        # Print die Median-Werte
        print(f'{titles[i]} - {timepoints[0]} (Original): {start_median:.2f}, (Filtered): {start_filtered_median:.2f}')
        print(f'{titles[i]} - {timepoints[1]} (Original): {middle_median:.2f}, (Filtered): {middle_filtered_median:.2f}')
        print(f'{titles[i]} - {timepoints[2]} (Original): {end_median:.2f}, (Filtered): {end_filtered_median:.2f}')

        # Plot für jeden Zeitpunkt
        axs[i, 0].plot(start_freq, start_power, label="Original Signal")
        axs[i, 0].plot(start_freq, start_filtered_power, label="Filtered Signal", linestyle='--')
        axs[i, 0].axvline(x=start_median, color='r', linestyle=':', label=f'Median (Original)')
        axs[i, 0].set_title(f'{titles[i]} - {timepoints[0]}')
        axs[i, 0].set_xlabel('Frequency (Hz)')
        axs[i, 0].set_ylabel('Power')
        axs[i, 0].legend()

        axs[i, 1].plot(middle_freq, middle_power, label="Original Signal")
        axs[i, 1].plot(middle_freq, middle_filtered_power, label="Filtered Signal", linestyle='--')
        axs[i, 1].set_title(f'{titles[i]} - {timepoints[1]}')
        axs[i, 1].set_xlabel('Frequency (Hz)')
        axs[i, 1].set_ylabel('Power')
        axs[i, 1].legend()

        axs[i, 2].plot(end_freq, end_power, label="Original Signal")
        axs[i, 2].plot(end_freq, end_filtered_power, label="Filtered Signal", linestyle='--')
        axs[i, 2].axvline(x=end_median, color='r', linestyle=':', label=f'Median (Original)')
        axs[i, 2].set_title(f'{titles[i]} - {timepoints[2]}')
        axs[i, 2].set_xlabel('Frequency (Hz)')
        axs[i, 2].set_ylabel('Power')
        axs[i, 2].legend()

    # Layout anpassen und anzeigen
    plt.tight_layout()
    plt.show()


def plot_median_frequency_changes(fatigue_1, fatigue_2, fatigue_3):
    """
    Erstellt einen Plot mit der Entwicklung der MEDIAN-WERTE DER POWER (Originaldaten)
    für Start, Middle und End jedes der drei Ermüdungstests.
    Dabei werden für jeden Test drei Datenpunkte (Start, Middle, End) durch eine Linie verbunden.
    """

    data_sets = [fatigue_1, fatigue_2, fatigue_3]
    titles = ["Fatigue 1", "Fatigue 2", "Fatigue 3"]

    # Liste für alle Medianwerte (jeweils 3 pro Test)
    all_medians = []

    for data in data_sets:
        # Datensatz in Start, Middle, End aufteilen
        start = data[:len(data) // 3]
        middle = data[len(data) // 3:2 * len(data) // 3]
        end = data[2 * len(data) // 3:]

        # Power-Spektrum für jedes Segment berechnen
        start_power, start_freq = get_power(start)  # get_power muss existieren
        middle_power, middle_freq = get_power(middle)
        end_power, end_freq = get_power(end)

        # Median des Power-Spektrums berechnen
        start_median = np.median(start_power)
        middle_median = np.median(middle_power)
        end_median = np.median(end_power)

        all_medians.append([start_median, middle_median, end_median])

    # Plot erstellen
    plt.figure(figsize=(6, 4))
    x_vals = [1, 2, 3]  # X-Achse für Start, Middle, End

    for i, medians in enumerate(all_medians):
        plt.plot(x_vals, medians, marker='o', label=titles[i])

    plt.xticks(x_vals, ["Start", "Middle", "End"])
    plt.xlabel("Test-Abschnitt")
    plt.ylabel("Median des Power-Spektrums")
    plt.title("Veränderung des Power-Medians über Start, Middle, End")
    plt.legend()
    plt.tight_layout()
    plt.show()



# Beispiel für das Laden der Daten und Entfernen des "A0:"-Präfixes
def load_and_process_data(file_path):
    df = pd.read_csv(file_path, header=None, names=["raw"])
    # Entferne das "A0:"-Präfix und konvertiere die Daten in Fließkommazahlen
    df["raw"] = df["raw"].astype(str).str.replace("A0:", "", regex=False)
    return df["raw"].astype(float).values