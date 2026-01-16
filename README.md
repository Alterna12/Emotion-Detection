# Emotion-Detection
Speech based emotions detection using MATLAB 

# Methodology
The emotion detection system is developed using a machine learning-based audio classification approach. The overall workflow consists of data preprocessing, feature extraction, model training, and evaluation.

1. Data Preprocessing
Audio files in .wav format are first loaded and standardized. If the audio is in stereo format, it is converted to mono by averaging the channels. The audio signal is then normalized to ensure consistent amplitude levels across all samples, reducing bias during training.

2. Feature Extraction
To represent emotional characteristics in speech, Mel-Frequency Cepstral Coefficients (MFCCs) are extracted from each audio file. MFCCs are chosen because they effectively capture human auditory perception and are widely used in speech and emotion recognition tasks. The mean value of MFCC features is computed for each audio sample to create a fixed-length feature vector.

3. Label Encoding
Emotion labels are derived from the dataset and converted into numerical form using label encoding. This allows the machine learning model to process categorical emotion classes efficiently.

4. Model Training
The extracted features are split into training and testing sets. A supervised machine learning classifier (such as Support Vector Machine or Random Forest) is trained using the training data. The model learns to associate MFCC feature patterns with corresponding emotion labels.

5. Model Evaluation
The trained model is evaluated using the testing dataset. Performance metrics such as accuracy, precision, recall, and F1-score are used to measure the effectiveness of the emotion classification system.

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
