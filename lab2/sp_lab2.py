import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy.io import wavfile

"""A1."""
# Function to compute the first derivative using finite difference method
def finite_difference(signal, sampling_rate):
    dt = 1.0 / sampling_rate
    derivative = np.diff(signal) / dt
    return np.concatenate(([0], derivative))

# Load the WAV file
file_path = 'lab1/Speech.mp3'
sampling_rate, signal = wavfile.read(file_path)

# Compute the first derivative
derivative = finite_difference(signal, sampling_rate)

# Plot the original signal and its derivative
time = np.arange(0, len(signal)) / sampling_rate

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, signal)
plt.title('Original Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
time_derivative = np.arange(0, len(derivative)) / sampling_rate
plt.plot(time_derivative, derivative)
plt.title('First Derivative')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

"""A3."""
def finite_difference(signal, sampling_rate):
    dt = 1.0 / sampling_rate
    derivative = np.diff(signal) / dt
    return np.concatenate(([0], derivative))

def zero_crossings(derivative):
    return np.where(np.diff(np.sign(derivative)))[0]

def calculate_average_length(zero_crossings):
    return np.mean(np.diff(zero_crossings))

# Load the WAV files for 5 speech signals
file_paths = [r'lab2\adi\word1.wav', r'lab2\adi\word2.wav', r'lab2\adi\word3.wav', r'lab2\adi\word4.wav', r'lab2\adi\word5.wav']
file_paths1 = [r'lab2\bag\word1.wav', r'lab2\bag\word2.wav', r'lab2\bag\word3.wav', r'lab2\bag\word4.wav', r'lab2\bag\word5.wav']

# Compare each pair of corresponding files
for i, (file_path, file_path1) in enumerate(zip(file_paths, file_paths1), 1):
    # Load the first WAV file
    sampling_rate, signal = wavfile.read(file_path)

    # Load the second WAV file
    _, signal1 = wavfile.read(file_path1)

    # Ensure both signals have the same length
    min_length = min(len(signal), len(signal1))
    signal = signal[:min_length]
    signal1 = signal1[:min_length]

    # Compute the first derivative for the first file
    derivative = finite_difference(signal, sampling_rate)

    # Detect zero crossings for the first file
    zero_crossings_indices = zero_crossings(derivative)

    # Calculate average length between zero crossings for speech for the first file
    speech_zero_crossings = zero_crossings_indices[zero_crossings_indices < len(signal) / 2]
    average_length_speech = calculate_average_length(speech_zero_crossings)

    # Plot the original signals for both files in the same graph
    time = np.arange(0, min_length) / sampling_rate

    plt.figure(figsize=(12, 4))
    plt.plot(time, signal, label=f'{file_path} - Original Signal')
    plt.plot(time, signal1, label=f'{file_path1} - Original Signal1')
    plt.title(f'Comparison: {file_path} vs {file_path1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f"File: {file_path}, Average length between zero crossings for speech: {average_length_speech} samples")
