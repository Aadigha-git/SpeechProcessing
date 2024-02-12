# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# import numpy as np
# import sounddevice as sd

# """A1"""
# # Load audio file
# audio_signal, sample_rate = librosa.load("lab1\Speech.mp3")

# # Compute first derivative of the audio signal
# first_derivative = np.diff(audio_signal)
# first_derivative /= np.max(np.abs(first_derivative))  # Normalize

# print("Playing First Derivative Signal:")

# # Plotting the first derivative
# plt.figure(figsize=(10, 5))
# librosa.display.waveshow(first_derivative, sr=sample_rate, color='red')
# plt.title('FIRST DERIVATIVE')
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Play the first derivative of the audio signal
# sd.play(first_derivative, rate=sample_rate)

# plt.figure(figsize=(10, 5))
# librosa.display.waveshow(audio_signal, sr=sample_rate)
# plt.title('Waveform')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# # Playing back the audio
# sd.play(audio_signal, rate=sample_rate)

# """A2"""

# zero_crossings = np.where(np.diff(np.sign(first_derivative)))[0]
# zero_crossing_diffs = np.diff(zero_crossings)
# threshold = 1000

# # Distinguish between speech and silence based on the threshold
# speech_intervals = zero_crossing_diffs[zero_crossing_diffs > threshold]
# silence_intervals = zero_crossing_diffs[zero_crossing_diffs <= threshold]

# # Calculate average lengths
# avg_speech_interval_length = np.mean(speech_intervals)
# avg_silence_interval_length = np.mean(silence_intervals)

# # Print average lengths
# print("Average length between consecutive zero crossings in speech regions:", avg_speech_interval_length)
# print("Average length between consecutive zero crossings in silence regions:", avg_silence_interval_length)

# # Plotting
# plt.figure(figsize=(10, 4))
# plt.plot(zero_crossing_diffs, label='All intervals', color='red')
# plt.plot(np.arange(len(speech_intervals)), speech_intervals, 'ro', label='Speech intervals', color='red')
# plt.plot(np.arange(len(speech_intervals), len(speech_intervals) + len(silence_intervals)), silence_intervals, 'bo', label='Silence intervals', color='green')
# plt.title('Pattern of Zero Crossings')
# plt.xlabel('Interval Index')
# plt.ylabel('Difference between Consecutive Zero Crossings')
# plt.legend()
# plt.show()


# """A3"""

# adi_wav = r'lab2\adi\word1.wav'
# bag_wav = r'lab2\bag\word1.wav'

# # Load audio signals
# y1, sr1 = librosa.load(adi_wav)
# y2, sr2 = librosa.load(bag_wav)

# # Duration of audio files
# duration1 = librosa.get_duration(y=y1, sr=sr1)
# duration2 = librosa.get_duration(y=y2, sr=sr2)

# print("The length Duration of first audio file:", duration1, "seconds")
# print("The length Duration of Second audio file:", duration2, "seconds")

# # Remove silence function
# def remove_silence(y, sr, threshold=0.01):
#     yt = librosa.effects.trim(y, top_db=threshold)
#     return yt[0]

# # Trim silence from audio signals
# adi_trim = remove_silence(y1, sr1)
# bag_trim = remove_silence(y2, sr2)

# # Calculate the time axes for the trimmed audio files
# time1 = np.linspace(0, len(adi_trim) / sr1, len(adi_trim))
# time2 = np.linspace(0, len(bag_trim) / sr2, len(bag_trim))

# # Plot the trimmed audio files
# plt.figure(figsize=(18, 6))
# plt.plot(time1, adi_trim, label='Adithya P audio')
# plt.plot(time2, bag_trim, label='Bharadwaj audio')
# plt.legend()
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Overlapped Audio Files')
# plt.show()


import librosa
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as ipd

# Load audio file
audio_signal, sample_rate = librosa.load(r"lab1\Speech.mp3")

# Compute first derivative of the audio signal
first_derivative = np.diff(audio_signal)
first_derivative /= np.max(np.abs(first_derivative))  # Normalize

# Plotting the first derivative
plt.figure(figsize=(10, 5))
plt.plot(first_derivative, color='red')
plt.title('FIRST DERIVATIVE')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()

# Play the first derivative of the audio signal
ipd.Audio(first_derivative, rate=sample_rate)

# Plotting the waveform
plt.figure(figsize=(10, 5))
plt.plot(audio_signal)
plt.title('Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()

# Playing back the audio
ipd.Audio(audio_signal, rate=sample_rate)