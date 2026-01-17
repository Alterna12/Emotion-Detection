# Emotion-Detection
Speech based emotions detection using MATLAB 

# Methodology
The emotion detection system is developed using a machine learning-based audio classification approach. The overall workflow consists of data preprocessing, feature extraction, model training, and evaluation.

1. Data Collection
  - Dataset: 535 WAV audio files
  - Emotions (7 classes): Anger, Boredom, Disgust, Fear, Happiness, Sadness, Neutral
  - Split: 80% training, 20% testing (hold-out validation)
2. Signal Preprocessing
  - Amplitude Normalization: Standardizes audio volume
  - Noise Reduction: Removes background noise using filtering
  - Framing: Splits audio into 25ms frames with 10ms overlap
  - Windowing: Applies Hamming window to each frame
3. Feature Extraction
  - Primary Feature: Mel-Frequency Cepstral Coefficients (MFCC)
  - MFCC Extraction Steps:
 1. Pre-emphasis Filtering: Amplifies high frequencies
 2. Hamming Windowing: Reduces spectral leakage
 3. FFT: Converts to frequency domain
 4. Mel Filter Bank: Applies perceptually-inspired scaling
 5. DCT: Produces final cepstral coefficients
Additional Features: Pitch, Energy, Formants
4. Data Balancing & Classification
4.1 Data Balancing:
- Technique: Synthetic Minority Over-sampling Technique (SMOTE)
- Purpose: Addresses class imbalance in training data
- Method: Generates synthetic samples for minority classes
4.2 Classification Model:
- Algorithm: Support Vector Machine (SVM)
- Kernel: Radial Basis Function (RBF)
- Multi-class Strategy: One-vs-All
4.3 Parameter Optimization:
- Method: Grid Search with 5-fold cross-validation
- Optimized Parameters:
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
The results show that audio-based emotion detection is feasible using MFCC features and supervised learning models. The classifier is able to recognize emotional patterns in speech with reasonable accuracy, indicating that MFCCs effectively capture emotional cues such as pitch, tone, and intensity.

Misclassification primarily occurs between emotions with similar acoustic characteristics, such as neutral and calm, or happy and excited. This suggests that emotional overlap in speech can affect classification performance. Increasing dataset size and incorporating additional features such as chroma or spectral contrast could further improve accuracy.

Overall, the system demonstrates good generalization on unseen data and validates the effectiveness of machine learning techniques for emotion detection. The results confirm that audio emotion recognition can be applied in real-world applications such as human-computer interaction, virtual assistants, and mental health monitoring systems.

# Code
1. Download SMOTE 
2. Use this code : https://github.com/Alterna12/Emotion-Detection/blob/main/Final_Projek_v5.m
3. Run this for the app in MATLAB : https://github.com/Alterna12/Emotion-Detection/blob/main/app1.mlapp
# PPT File 
https://github.com/Alterna12/Emotion-Detection/blob/main/PPT%20Final%20Project%20Emotion%20Detection.pdf
