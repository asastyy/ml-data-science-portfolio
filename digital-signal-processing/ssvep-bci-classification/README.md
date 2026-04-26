# SSVEP BCI Classification

EEG-based target classification for a Steady-State Visual Evoked Potential (SSVEP) brain-computer interface.

The project compares frequency-recognition methods for SSVEP signals and evaluates how decision window length affects classification accuracy.

## Overview

SSVEP-based BCI systems classify a visual target by detecting frequency-specific responses in EEG signals.

This project implements and compares:

- FFT-based frequency detection
- Canonical Correlation Analysis (CCA)
- Filter Bank Canonical Correlation Analysis (FBCCA)
- ICA-based eye artifact removal

The analysis is performed across multiple subjects and decision window sizes from **0.5 to 5.0 seconds**.

## Dataset

The dataset contains EEG recordings from SSVEP stimulation experiments.

Each subject includes:

- 64 EEG channels
- 250 Hz sampling frequency
- 40 stimulation frequencies
- 6 trials per frequency
- 6-second EEG epochs

The evaluation focuses on **16 selected target frequencies**.

## Methods

- EEG preprocessing
- Occipital and parietal channel selection
- Frequency-domain analysis
- Fast Fourier Transform
- Canonical Correlation Analysis
- Filter Bank CCA
- Independent Component Analysis
- Accuracy evaluation
- Confusion matrix analysis

## Results

FBCCA achieved the best overall performance across all tested decision windows.

Average median accuracy across window sizes:

| Method | Average Median Accuracy |
|---|---:|
| FFT | 0.539 |
| CCA | 0.773 |
| FBCCA | 0.842 |

FBCCA reached **1.000 median accuracy** for decision windows of **4.0 seconds and longer**.  
CCA also performed strongly on longer windows, while FFT served as a simple and interpretable baseline.

ICA reduced frontal eye-movement artifacts, although its effect on classification accuracy was mixed and depended on the method and window size.

## Tech Stack

**Language:** Python  
**Signal Processing:** NumPy, SciPy  
**EEG Processing:** MNE  
**Machine Learning / Evaluation:** scikit-learn  
**Data Analysis:** pandas  
**Visualization:** Matplotlib  
**Environment:** Jupyter Notebook
