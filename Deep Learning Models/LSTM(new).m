% Load and prepare data
data = readtable('C:\Users\HP\Downloads\nutrient_0.01 (1).csv');
time = data.Time;
population = data.Population;

% Normalize data
pop_min = min(population);
pop_max = max(population);
pop_normalized = (population - pop_min) / (pop_max - pop_min);

% Check for NaNs
if any(isnan(pop_normalized))
    error('NaNs detected in pop_normalized. Please clean the data.');
end

% Add small noise to training data only (avoid test data leakage)
noise = 0.01 * randn(size(pop_normalized)); % Small noise (0.01 scale)
pop_normalized_noisy = pop_normalized + noise;

% Create sequences (use noisy data for training)
sequenceLength = 5;
XTrain = {};
YTrain = {};
for i = 1:length(pop_normalized_noisy) - sequenceLength
    XTrain{i} = pop_normalized_noisy(i:i+sequenceLength-1)'; % [5, 1]
    YTrain{i} = pop_normalized(i+sequenceLength);             % Use original for target
end

% Split data (60% train, 10% val, 30% test)
numSamples = length(XTrain);
numTrain = floor(0.6 * numSamples);
numVal = floor(0.1 * numSamples);
XTrainSplit = XTrain(1:numTrain);
YTrainSplit = reshape(cell2mat(YTrain(1:numTrain)), [], 1);
XValSplit = XTrain(numTrain+1:numTrain+numVal);
YValSplit = reshape(cell2mat(YTrain(numTrain+1:numTrain+numVal)), [], 1);
XTestSplit = XTrain(numTrain+numVal+1:end);
YTestSplit = reshape(cell2mat(YTrain(numTrain+numVal+1:end)), [], 1);

% Verify sizes
fprintf('XTrainSplit: %d sequences\n', length(XTrainSplit));
fprintf('YTrainSplit: %d responses\n', length(YTrainSplit));
fprintf('XValSplit: %d sequences\n', length(XValSplit));
fprintf('YValSplit: %d responses\n', length(YValSplit));
fprintf('XTestSplit: %d sequences\n', length(XTestSplit));
fprintf('YTestSplit: %d responses\n', length(YTestSplit));
disp('YTrainSplit shape: '); disp(size(YTrainSplit));
disp('YValSplit shape: '); disp(size(YValSplit));
disp('YTestSplit shape: '); disp(size(YTestSplit));

% Define LSTM architecture
layers = [
    sequenceInputLayer(1)
    lstmLayer(150, 'OutputMode', 'last')
    dropoutLayer(0.2) 
    fullyConnectedLayer(100)
    dropoutLayer(0.2)
    fullyConnectedLayer(50)
    dropoutLayer(0.1)
    fullyConnectedLayer(1)
    regressionLayer
];

% Training options
options = trainingOptions('adam', ...
    'MaxEpochs', 150, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 100, ...
    'L2Regularization', 0.0005, ...
    'Shuffle', 'never', ...
    'ValidationData', {XValSplit, YValSplit}, ...
    'ValidationFrequency', 30, ...
    'ValidationPatience',25, ...
    'Verbose', 1, ...
    'Plots', 'training-progress');

% Set fixed start point for same RMSE every time
rng(1);

% Train the model with runtime measurement
disp('Training LSTM model...');
tic;
net = trainNetwork(XTrainSplit, YTrainSplit, layers, options);
lstm_time = toc;
fprintf('LSTM training time: %.2f seconds\n', lstm_time);

% Test set prediction
YTestPred = predict(net, XTestSplit);
YTestActual = YTestSplit * (pop_max - pop_min) + pop_min;
YTestPredDenorm = YTestPred * (pop_max - pop_min) + pop_min;

% Calculate RMSE
rmse = sqrt(mean((YTestActual - YTestPredDenorm).^2));
fprintf('Test RMSE: %.4f\n', rmse);

% Plot test predictions
testStartIdx = numTrain + numVal + sequenceLength + 1;
testTime = time(testStartIdx:end);
figure;
plot(time, population, 'b-');
hold on;
plot(testTime, YTestPredDenorm, 'r--');
legend;
xlabel('Time');
ylabel('Population');
title('Bacterial Growth: Test Set Prediction');

% Test set prediction
YTestPred = predict(net, XTestSplit);
YTestActual = YTestSplit * (pop_max - pop_min) + pop_min;
YTestPredDenorm = YTestPred(:, 1) * (pop_max - pop_min) + pop_min;
YTestPredDenorm = YTestPredDenorm(:);

% Calculate RMSE, MAE, and R² for test set
rmse_test = sqrt(mean((YTestActual - YTestPredDenorm).^2));
mae_test = mean(abs(YTestActual - YTestPredDenorm));
yMean = mean(YTestActual);
ssTot = sum((YTestActual - yMean).^2);
ssRes = sum((YTestActual - YTestPredDenorm).^2);
rSquared_test = 1 - (ssRes / ssTot);
fprintf('Test RMSE: %.4f\n', rmse_test);
fprintf('Test MAE: %.4f\n', mae_test);
fprintf('Test R²: %.4f\n', rSquared_test);

% Train set prediction
YTrainPred = predict(net, XTrainSplit);
YTrainActual = YTrainSplit * (pop_max - pop_min) + pop_min;
YTrainPredDenorm = YTrainPred(:, 1) * (pop_max - pop_min) + pop_min;
YTrainPredDenorm = YTrainPredDenorm(:);

% Calculate RMSE, MAE, and R² for train set
rmse_train = sqrt(mean((YTrainActual - YTrainPredDenorm).^2));
mae_train = mean(abs(YTrainActual - YTrainPredDenorm));
yMeanTrain = mean(YTrainActual);
ssTotTrain = sum((YTrainActual - yMeanTrain).^2);
ssResTrain = sum((YTrainActual - YTrainPredDenorm).^2);
rSquared_train = 1 - (ssResTrain / ssTotTrain);

% Print train metrics
fprintf('Train RMSE: %.4f\n', rmse_train);
fprintf('Train MAE: %.4f\n', mae_train);
fprintf('Train R²: %.4f\n', rSquared_train);

% Save model
save('bacterial_growth_lstm.mat', 'net');

% Save test and train metrics with runtime
lstm_metrics = [rmse_test, mae_test, rSquared_test, rmse_train, mae_train, rSquared_train, lstm_time];
save('lstm_metrics.mat', 'lstm_metrics');
