import librosa
import numpy as np
import soundfile as sf
import IPython.display as ipd
import pandas as pd
import matplotlib.pyplot as plt

# A1
audio_path = "lab1\Speech.mp3"
y, sr = librosa.load(audio_path, sr=None)

y_trim, _ = librosa.effects.trim(y)
trimmed_audio_path = "trimmed_speech.wav"
sf.write(trimmed_audio_path, y_trim, sr)

original_audio, _ = librosa.load(audio_path, sr=None)
trimmed_audio, _ = librosa.load(trimmed_audio_path, sr=None)

print("Original Audio:")
sf.write("original_audio.wav", original_audio, sr)
print("Trimmed Audio:")
sf.write("trimmed_audio.wav", trimmed_audio, sr)

# A2
y, sr = librosa.load(audio_path, sr=None)

# Split the audio based on detected silences
split_intervals = librosa.effects.split(y, top_db=30)  # Adjust top_db as needed

# Save each split interval as a separate audio file and display
for i, (start, end) in enumerate(split_intervals):
    split_audio = y[start:end]
    split_audio_path = f"split_{i}.wav"
    sf.write(split_audio_path, split_audio, sr)
    print(f"Split Audio {i}: {split_audio_path}")

"""A3."""
def calculate_continuous_average_energy(signal, sampling_rate):
    window_length = int(0.05 * sampling_rate)  # 50 ms window length
    energy = np.square(signal)
    Ek = np.convolve(energy, np.ones(window_length)/window_length, mode='valid')
    return Ek

def compute_normalized_energy(energy):
    avr_Ek = np.mean(energy)
    var_Ek = np.var(energy)
    Em = (energy - avr_Ek) / var_Ek
    return Em

def identify_lobes_maxima_indices(energy):
    zero_crossings = np.where(np.diff(np.sign(energy)))[0]
    maxima_indices = zero_crossings[1:-1]  # Exclude the first and last zero crossings
    maxima = energy[maxima_indices]
    return maxima, maxima_indices

# Load the audio signal
file_path = "lab1\Speech.mp3"
signal, sampling_rate = librosa.load(file_path, sr=None)

# Step 1: Calculate continuous average energy
Ek = calculate_continuous_average_energy(signal, sampling_rate)

# Step 2: Compute normalized continuous average energy
Em = compute_normalized_energy(Ek)

# Step 3: Identify maxima and indices of the lobes
maxima, maxima_indices = identify_lobes_maxima_indices(Em)

time = np.arange(len(Em)) * 0.05  # Time vector (50 ms steps)
plt.figure(figsize=(10, 5))
plt.plot(time, Em, label='Normalized Energy')
plt.plot(maxima_indices * 0.05, maxima, 'ro', label='Maxima Indices')
plt.xlabel('Time (s)')
plt.ylabel('Normalized Energy')
plt.title('Normalized Energy with Maxima Indices')
plt.legend()
plt.grid(True)
plt.show()