clc; clear; close all;

%% === 1. Path Dataset ===
datasetPath = 'C:\Users\alexn\OneDrive\Documents\MATLAB\wav';
audioFiles = dir(fullfile(datasetPath, '*.wav'));

if isempty(audioFiles)
    error('Tidak ditemukan file .wav di folder yang ditentukan.');
end
fprintf('Total file audio ditemukan: %d\n', length(audioFiles));

%% === 2. Persiapan variabel ===
features = [];
labels = {};
skippedFiles = {};

%% === 3. Looping semua file audio ===
for i = 1:length(audioFiles)
    filename = audioFiles(i).name;
    filepath = fullfile(datasetPath, filename);

    % --- Load audio ---
    [y, fs] = audioread(filepath);
    if size(y,2) > 1
        y = mean(y,2);
    end
    y = y / (max(abs(y)) + eps);

    if max(abs(y)) < 0.001
        skippedFiles{end+1} = filename;
        continue;
    end

    % --- Ekstraksi MFCC dengan v_melcepst (ASLI training Anda) ---
    n_mfcc = 13;
    win = round(0.025 * fs);
    hop = round(0.010 * fs);

    try
        coeffs = v_melcepst(y, fs, '0', n_mfcc, 20, win, hop);
    catch
        skippedFiles{end+1} = filename;
        continue;
    end

    if isempty(coeffs)
        skippedFiles{end+1} = filename;
        continue;
    end

    % Delta MFCC + Delta-Delta
    deltaMFCC  = diff([zeros(1,size(coeffs,2)); coeffs]);
    deltaDelta = diff([zeros(1,size(deltaMFCC,2)); deltaMFCC]);

    % === SPECTRAL FEATURES (ASLI training Anda) ===
    N = length(y);
    Y = abs(fft(y));
    Y = Y(1:floor(N/2));
    freqs = linspace(0, fs/2, length(Y));

    spectralCentroid  = sum(freqs .* Y') / (sum(Y) + eps);
    spectralBandwidth = sqrt(sum((freqs - spectralCentroid).^2 .* Y') / (sum(Y) + eps));

    threshold = 0.85 * sum(Y);
    idx = find(cumsum(Y) >= threshold, 1, 'first');
    spectralRollOff = freqs(idx);

    energy = mean(y.^2);
    zeroCrossings = sum(abs(diff(sign(y)))) / length(y);

    try
        f0 = pitch(y, fs, "Method", "PEF", "Range", [50 400]);
        pitchMean = mean(f0);
        if isnan(pitchMean), pitchMean = 0; end
    catch
        pitchMean = 0;
    end

    % === GABUNGKAN FITUR SESUAI TRAINING ===
    feat = [ ...
        mean(coeffs), std(coeffs), ...
        mean(deltaMFCC), std(deltaMFCC), ...
        mean(deltaDelta), std(deltaDelta), ...
        energy, zeroCrossings, pitchMean, ...
        spectralCentroid, spectralBandwidth, spectralRollOff ...
    ];

    % === LABEL BERDASARKAN KARAKTER 6 ===
    emoCode = filename(6);
    switch emoCode
        case 'W', label = 'anger';
        case 'L', label = 'boredom';
        case 'E', label = 'disgust';
        case 'A', label = 'fear';
        case 'F', label = 'happiness';
        case 'T', label = 'sadness';
        case 'N', label = 'neutral';
        otherwise, label = 'unknown';
    end

    features = [features; feat];
    labels = [labels; label];
end

labels = categorical(labels);

%% === 🔧 FIXED NORMALIZATION — Single source of truth ===
fprintf("\nApplying NORMALIZE() once (same as original pipeline)...\n");
[features, mu, sigma] = normalize(features);  
% Now mu & sigma correspond to the normalized features (MATCHES detection)

%% === 4. SMOTE on NORMALIZED features ===
fprintf("\nMelakukan SMOTE Oversampling...\n");
[X_smote, Y_smote] = smote(features, labels, 5);

%% === 5. Grid Search SVM RBF ===
fprintf("\nGrid Search SVM RBF (KFold=5)...\n");

kernelScaleRange = 1:1:10;
boxRange = 1:1:10;

bestAcc = 0;
bestKS = 1;
bestC = 1;

for ks = kernelScaleRange
    for C = boxRange

        t = templateSVM('KernelFunction','rbf', ...
                        'KernelScale', ks, ...
                        'BoxConstraint', C, ...
                        'Standardize', true);   % ← keep this

        Mdl = fitcecoc(X_smote, Y_smote, 'Learners', t, 'Coding','onevsall');
        CVMdl = crossval(Mdl, 'KFold', 5);
        acc = 1 - kfoldLoss(CVMdl);

        if acc > bestAcc
            bestAcc = acc;
            bestKS = ks;
            bestC = C;
        end
    end
end

fprintf('>> Best KernelScale = %d\n', bestKS);
fprintf('>> Best BoxConstraint = %d\n', bestC);
fprintf('>> Best CV Accuracy = %.2f%%\n', bestAcc*100);

%% === 6. Split Data (80/20) ===
cv = cvpartition(Y_smote, "HoldOut", 0.2);

XTrain = X_smote(training(cv), :);
YTrain = Y_smote(training(cv));
XTest  = X_smote(test(cv), :);
YTest  = Y_smote(test(cv), :);

%% === 7. Train Final Model ===
tFinal = templateSVM('KernelFunction','rbf', ...
                     'KernelScale', bestKS, ...
                     'BoxConstraint', bestC, ...
                     'Standardize', true);

SVMModel = fitcecoc(XTrain, YTrain, ...
                    'Learners', tFinal, ...
                    'Coding', 'onevsall');

%% === 8. Evaluation ===
YPred = predict(SVMModel, XTest);
accuracy = sum(YPred == YTest) / numel(YTest) * 100;

fprintf("\nAkurasi Testing (20%% Holdout) = %.2f%%\n", accuracy);

figure;
confusionchart(YTest, YPred);
title(sprintf('Confusion Matrix (Akurasi: %.2f%%)', accuracy));

%% === SAVE MODEL + CONSISTENT NORMALIZATION (vital for detectEmotion) ===
save('emotion_svm_model.mat', 'SVMModel', 'mu', 'sigma');
fprintf('\nModel and normalization stats saved to emotion_svm_model.mat\n');