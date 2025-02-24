# AI in Speech Processing

## Overview
This coursework focuses on utilizing Librosa, a Python package for audio and music analysis, to explore fundamental aspects of speech processing.

## Lab 1 key learnings: Librosa
Lab 1 covers the following key areas:
- **Usage of Librosa:**
  - Loading and handling audio files.
  - Exploring functionalities for audio signal processing.
- **Visualization:**
  - Visualizing audio signals for better understanding.
- **Feature Extraction:**
  - Extracting essential features from audio data.
- **Effects Creation:**
  - Applying effects such as noise removal and segmentation.
- **Sampling:**
  - Understanding and working with audio sampling.

 ## Lab 2 Key Learnings: Signal Derivatives and Zero Crossings
Lab 2 introduces the following concepts:
- **Finite Difference Method:**
  - Calculation of the first derivative of the speech signal using the finite difference method, which helps in understanding signal dynamics.
- **Zero Crossing Detection:**
  - Identifying zero crossings in the signal to analyze speech characteristics, like frequency and signal stability.
- **Comparison of Signals:**
  - Comparing multiple speech signals and extracting features such as the average length between zero crossings to gain insights into speech patterns.
- **Visualization:**
  - Plotting both the original signals and their derivatives to visualize changes over time.

## Lab 3 Key Learnings: Audio Trimming, Splitting, and Energy Analysis

Lab 3 focuses on manipulating audio signals by trimming, splitting, and analyzing their energy characteristics.

### A1: Audio Trimming
- **Trimming Silent Parts:**
  - The audio signal is trimmed to remove silent parts using `librosa.effects.trim`.
  - The trimmed and original audio signals are saved as separate files.
  
### A2: Audio Splitting
- **Splitting Based on Silence:**
  - The audio is split into non-silent segments based on energy thresholding using `librosa.effects.split`.
  - Each non-silent segment is saved as an individual audio file for further processing.

### A3: Energy Analysis
- **Continuous Average Energy Calculation:**
  - The continuous average energy is calculated over a sliding window (50 ms).
  - This helps identify regions of high energy, which might correspond to speech or other significant audio events.
  
- **Normalized Energy Computation:**
  - The energy is normalized to detect the significant peaks (lobes) in the audio signal.

- **Lobe Maxima Identification:**
  - The local maxima (peaks) in the energy signal are identified, providing insight into the structure of the audio signal.

## Lab 4 Key Learnings: Spectral Analysis and Visualization

Lab 4 introduces methods for analyzing the frequency content of speech signals using various spectral techniques.

### A1: Spectral Analysis (FFT)
- **Fourier Transform:**
  - The Fast Fourier Transform (FFT) is applied to the speech signal to analyze its frequency content.
  - The amplitude spectrum is visualized, showing how the signal's energy is distributed across different frequencies.

### A2: Inverse Fourier Transform
- **Reconstruction from Frequency Domain:**
  - The Inverse FFT is applied to reconstruct the original signal from its frequency-domain representation.
  - The result is plotted to visualize how the signal looks in the time domain after inverse transformation.

### A3: Spectral Analysis of a Word
- **Word-Specific Spectral Analysis:**
  - A specific word segment is isolated from the speech signal (from 2 to 3 seconds).
  - The FFT is applied to this segment to examine its frequency content.
  - The amplitude spectrum of this word is visualized to explore the frequency characteristics of spoken words.

### A4: Spectral Analysis with Rectangular Window
- **Windowed FFT Analysis:**
  - A rectangular window of a fixed length is applied to a segment of the signal.
  - The amplitude spectrum of the windowed signal is computed and visualized to see how frequency components change within the selected segment.

### A5: Break into Windows and Create Heatmap
- **Sliding Window Spectrogram:**
  - The signal is broken into overlapping windows (20 ms window, 10 ms hop length).
  - A heatmap (spectrogram) is generated to visualize how the frequency content of the signal changes over time.
  - The heatmap is displayed with a logarithmic scale for better visualization of low-amplitude components.

### A6: Spectrogram Using Scipy
- **Spectrogram Calculation with Scipy:**
  - The `scipy.signal.spectrogram` function is used to compute the spectrogram of the speech signal.
  - The result is visualized with a color-mapped plot showing the frequency and intensity of the signal over time.

### Code Breakdown:
- **FFT & IFFT:** Analyzes the frequency content of the signal and reconstructs it from the frequency domain.
- **Windowed FFT & Spectrogram:** Examines the frequency content over specific time segments, with visualization as a heatmap.
- **Scipy Spectrogram:** Generates a spectrogram using a built-in function, highlighting the intensity variations across frequencies over time.

## Lab 5 Key Learnings: Spectral Filtering and Visualization

Lab 5 explores the use of different filters on speech signals to modify their frequency content. This includes applying various types of filters and visualizing the effects on the signal.
- **Bandpass Filter:**
  - A bandpass filter is applied to the speech signal, allowing frequencies between 1000 Hz and 4000 Hz to pass through.
  - The filtered signal is visualized and saved as a WAV file.

- **Highpass Filter:**
  - A highpass filter is applied to the speech signal, allowing frequencies above 3000 Hz to pass through.
  - The filtered signal is visualized and saved as a WAV file.

- **Cosine Filter:**
  - A cosine window is applied to the signal, creating a cosine-shaped frequency response for filtering.
  - The filtered signal is visualized and saved as a WAV file.

- **Gaussian Filter:**
  - A Gaussian window is applied to the signal, smoothing the frequency response with a Gaussian-shaped filter.
  - The filtered signal is visualized and saved as a WAV file.

### Code Breakdown:
- **Spectral Analysis:** Uses FFT to analyze and visualize the frequency content of the signal.
- **Filtering:** Various filters (rectangular, bandpass, highpass, cosine, and Gaussian) are applied to modify the frequency content of the speech signal.
- **Visualization:** Both the time-domain waveforms and frequency-domain spectral components are visualized to understand the impact of the filters.

## Lab 6 Key Learnings: Frequency Domain Analysis of Speech Sounds

Lab 6 focuses on analyzing speech signals in the frequency domain, specifically through the use of the Fast Fourier Transform (FFT) to observe the spectral properties of voiced and non-voiced sounds, as well as consonant snippets.

### Code Breakdown:
- **Fourier Transform:** FFT is applied to both non-voiced and consonant snippets.
- **Frequency Bins Calculation:** Frequency bins are computed using `np.fft.fftfreq`, which maps each FFT result to its corresponding frequency value.
- **Amplitude Spectrum Plotting:** The magnitude of the FFT results is plotted to analyze the frequency content of the signal.

## Lab 7: Hidden Markov Model (HMM) for Speech Signal Classification

In this lab, we explore how to use a Hidden Markov Model (HMM) to classify speech signals and predict state sequences. The HMM is trained using features extracted from the Short-Time Fourier Transform (STFT) of speech signals.

# Lab 8: Speech Recognition and Feature Extraction

## Objective
1. Perform feature extraction from audio signals.
2. Build and train an LSTM-based model for speech recognition.
3. Synthesize speech signals using phoneme sequences.
4. Analyze and visualize the extracted features and the results.


## Dependencies
- Python
- Librosa
- Matplotlib
- Numpy
- Sounddevice (for optional audio playback)
- Scipy
