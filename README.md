# VibraHybrid-FD: A Hybrid STFT–HHT–Boruta Python Toolkit

Software Architecture & Functional Modules

The repository is structured to reflect the incremental development of the proposed diagnostic framework, allowing for comprehensive comparative analysis:

- data/: Contains the standardized vibration dataset (processed_data.csv) used for model training and validation.
- src/Preprocessing_data.py: A dedicated module for signal synchronization, data cleaning, and class labeling (Labels 1–9).
- src/Fan_Machine_Learning.py: Represents the Baseline Pipeline, where raw temporal signals are directly integrated with machine learning classifiers without advanced feature engineering.
- src/Fan_STFT_Machine_Learning.py: Implements the Spectral-Based Pipeline, utilizing Short-Time Fourier Transform (STFT) for frequency-domain feature extraction prior to classification.
- src/Fan_STFT_Boruta_Machine_Learning.py: The Proposed Optimized Pipeline. This module integrates STFT/HHT hybrid feature extraction with the Boruta algorithm for all-relevant feature selection. It constitutes the core contribution of this work, achieving an optimal accuracy of 99.13%.

Key Technical Features
- Hybrid Signal Processing: Integration of linear (STFT) and non-linear (HHT/EMD) decomposition techniques.
- Feature Optimization: Automated identification of the most significant diagnostic descriptors using Boruta.
- Explainable AI (XAI): Post-hoc model interpretation via SHAP (SHapley Additive exPlanations) to ensure physical consistency in fault identification.

Installation & Reproducibility
To ensure computational reproducibility, please initialize the environment as follows:

1. Dependency Installation:
pip install -r requirements.txt

2. Execution:
To reproduce the primary experimental results, execute the optimized pipeline:
python src/Fan_STFT_Boruta_Machine_Learning.py