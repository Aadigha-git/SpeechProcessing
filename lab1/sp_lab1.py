import librosa
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

def plot_waveform(y, sr):
    # Plots the waveform of the audio signal.
    """
    Parameters: -   y: Audio signal ndarray 
                    sr: Sampling rate
    """
    plt.figure(figsize=(10, 5))
    plt.plot(librosa.times_like(y), y)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform of the Speech Signal")
    plt.show()

def calculate_length_and_magnitude(y, sr):
    
    # Calculates and prints the length and magnitude range.
   
    signal_length_seconds = len(y) / sr
    magnitude_range = np.abs(y).max() - np.abs(y).min()

    print("Length of the signal:", signal_length_seconds, "seconds")
    print("Magnitude range of the signal:", magnitude_range)

def play_and_plot_segment(start_time, end_time, y, sr):
    # Plays and plots a segment of the audio signal
    """
    Parameters: - start_time: Start time of the segment in seconds
                - end_time: End time of the segment in seconds
    """
    segment = y[int(start_time * sr):int(end_time * sr)]

    # Play the segment
    sd.play(segment, sr)
    sd.wait()

    # Plot the segment
    time = np.linspace(start_time, end_time, len(segment))
    plt.plot(time, segment)
    plt.title(f'Audio Segment: {start_time} to {end_time} seconds')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.show()

# A1. Load audio file
filename = r"C:/Users/AADITYA/Desktop/sem6/SpeechProcessing/lab1/Speech.mp3"
y, sr = librosa.load(filename)

# Plot waveform
plot_waveform(y, sr)

# A2. Calculate length and magnitude range
calculate_length_and_magnitude(y, sr)

# A3 & A4. Play and plot different segments
play_and_plot_segment(2, 6, y, sr)
play_and_plot_segment(0, 2, y, sr)
play_and_plot_segment(1, 3, y, sr)
play_and_plot_segment(2, 4, y, sr)
