# Emotion-Detection
Speech based emotions detection using MATLAB 

# Dataset
[EmoDB](https://www.kaggle.com/datasets/piyushagni5/berlin-database-of-emotional-speech-emodb)

# Methodology
The emotion detection system is developed using a machine learning-based audio classification approach. The overall workflow consists of data preprocessing, feature extraction, model training, and evaluation.

1. Data Collection

    Dataset: 535 WAV audio files

    Emotions (7 classes): Anger, Boredom, Disgust, Fear, Happiness, Sadness, Neutral

    Split: 80% training, 20% testing (hold-out validation)

2. Signal Preprocessing

    Amplitude Normalization: Standardizes audio volume

    Noise Reduction: Removes background noise using filtering

    Framing: Splits audio into 25ms frames with 10ms overlap

    Windowing: Applies Hamming window to each frame

3. Feature Extraction

Primary Feature: Mel-Frequency Cepstral Coefficients (MFCC)

MFCC Extraction Steps:

    Pre-emphasis Filtering: Amplifies high frequencies

    Hamming Windowing: Reduces spectral leakage

    FFT: Converts to frequency domain

    Mel Filter Bank: Applies perceptually-inspired scaling

    DCT: Produces final cepstral coefficients

Additional Features: Pitch, Energy, Formants
4. Data Balancing & Classification
Data Balancing:

    Technique: Synthetic Minority Over-sampling Technique (SMOTE)

    Purpose: Addresses class imbalance in training data

    Method: Generates synthetic samples for minority classes

Classification Model:

    Algorithm: Support Vector Machine (SVM)

    Kernel: Radial Basis Function (RBF)

    Multi-class Strategy: One-vs-All

Parameter Optimization:

    Method: Grid Search with 5-fold cross-validation

    Optimized Parameters:

        KernelScale: 9

        BoxConstraint: 9

    CV Accuracy: 94.15%

5. Evaluation Metrics

The final model is evaluated using:

    Accuracy: Overall classification correctness

    Precision: Reliability of positive predictions

    Recall: Completeness of positive predictions

    F1-score: Harmonic mean of precision and recall

    Confusion Matrix: Detailed class-wise performance

Key Implementation Notes

    Platform: MATLAB R2021a or later

    Toolboxes Used: Signal Processing, Statistics and Machine Learning

    Key MATLAB Functions: mfcc(), fitcsvm(), gridsearch()

    Processing Time: ~15 minutes for full pipeline (may vary with hardware)

Expected Output

    Testing Accuracy: 94.92%

    Precision: 0.9558

    Recall: 0.9488

    F1-score: 0.9498

# Analysis

Report : https://github.com/Alterna12/Emotion-Detection/blob/main/Laporan%20Emotion%20Detector.pdf

The results confirm that audio-based emotion detection is highly effective using MFCC features and SVM classification. The classifier achieves excellent accuracy (94.92%), demonstrating that MFCCs effectively capture emotional cues related to pitch, tone, and spectral energy in speech.

Misclassification primarily occurs between emotions with similar acoustic characteristics, such as happiness and anger (both high-energy states) or disgust and fear (sharing certain spectral patterns). This indicates that emotional overlap in speech can pose a challenge even for a well-performing system.

The use of SMOTE for class balancing and grid search for hyperparameter tuning contributed to robust generalization on unseen data. The high precision (95.58%) and recall (94.88%) further validate the reliability of the machine learning pipeline.

Overall, the system not only demonstrates the feasibility of speech emotion recognition but also achieves performance suitable for real-world applications such as human-computer interaction, virtual assistants, and mental health monitoring—validating MFCC and SVM as a strong, practical approach to the task.

# Code
1. Download SMOTE 
2. Use this code : https://github.com/Alterna12/Emotion-Detection/blob/main/Final_Projek_v5.m
3. Run this for the app in MATLAB : https://github.com/Alterna12/Emotion-Detection/blob/main/app1.mlapp
# PPT File 
https://github.com/Alterna12/Emotion-Detection/blob/main/PPT%20Final%20Project%20Emotion%20Detection.pdf
