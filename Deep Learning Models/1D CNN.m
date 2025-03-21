%% Load Dataset
filename = 'C:\Users\HP\Downloads\nutrient_0.01 (1).csv';
dataTable = readtable(filename);

rng(42); % Set random seed

% Extract time and population
time = dataTable{:,1};
population = dataTable{:,2};

%% Visualize Raw Data
figure;
subplot(2,1,1);
plot(time, population, 'b.');
xlabel('Time'); ylabel('Population');
title('Raw Data: Population vs Time');
grid on;

%% Transform Data (Avoid Normalizing Time)
population = log(population + 1); % Log-transform population for stability
population = normalize(population, 'zscore'); % Standardize population

%% Visualize Transformed Data
subplot(2,1,2);
plot(time, population, 'r.');
xlabel('Time'); ylabel('Normalized Population');
title('Transformed Data: Log-Scaled and Normalized Population');
grid on;

%% Add Noise to Population Data
noiseFactor = 0.01; % Reduced noise intensity
population = population + noiseFactor * randn(size(population));

%% Shuffle and Split Data (Train: 70%, Validation: 10%, Test: 20%)
shuffledIdx = randperm(length(time));
time = time(shuffledIdx);
population = population(shuffledIdx);

n = length(time);
train_idx = 1:round(0.7*n);
val_idx = round(0.7*n)+1:round(0.8*n);
test_idx = round(0.8*n)+1:n;

XTrain = time(train_idx); YTrain = population(train_idx);
XVal = time(val_idx); YVal = population(val_idx);
XTest = time(test_idx); YTest = population(test_idx);

%% Design Simplified 1D CNN Model
layers = [ ...
    sequenceInputLayer(1, "Name", "Input"), ...
    convolution1dLayer(5, 16, "Padding", "same", "Name", "Conv1"), ...
    batchNormalizationLayer("Name", "BatchNorm1"), ...
    reluLayer("Name", "ReLU1"), ...
    dropoutLayer(0.3, "Name", "Dropout1"), ...
    convolution1dLayer(5, 32, "Padding", "same", "Name", "Conv2"), ...
    batchNormalizationLayer("Name", "BatchNorm2"), ...
    reluLayer("Name", "ReLU2"), ...
    dropoutLayer(0.3, "Name", "Dropout2"), ...
    fullyConnectedLayer(32, "Name", "FC1"), ...
    reluLayer("Name", "ReLU3"), ...
    fullyConnectedLayer(1, "Name", "FC2"), ...
    regressionLayer("Name", "Output") ...
    ];

%% Specify Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 500, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.0005, ...
    'ValidationData', {XVal', YVal'}, ...
    'Plots', 'training-progress', ...
    'Verbose', false, ...
    'L2Regularization', 0.001, ...
    'ValidationFrequency', 10, ...
    'Shuffle', 'every-epoch', ...
    'GradientThreshold', 1);

%% Train the Model
net = trainNetwork(XTrain', YTrain', layers, options);

%% Test the Model
YPred = predict(net, XTest');

%% Compute Metrics
mseError = mean((YTest - YPred).^2, 'omitnan'); % Mean Squared Error
rSquared = 1 - sum((YTest - YPred).^2) / sum((YTest - mean(YTest)).^2); % R-squared
fprintf('Mean Squared Error: %.4f\n', mseError);
fprintf('R-squared: %.4f\n', rSquared);

%% Plot Actual vs Predicted
figure;
plot(XTest, YTest, 'b', 'LineWidth', 1.5);
hold on;
plot(XTest, YPred, 'r--', 'LineWidth', 1.5);
legend('Actual', 'Predicted');
xlabel('Time'); ylabel('Population');
title('Actual vs Predicted Population Growth (Enhanced 1D CNN Model)');
grid on;
