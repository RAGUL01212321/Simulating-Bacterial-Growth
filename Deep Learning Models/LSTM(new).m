clc; clear; close all;

rng(21); % Set a fixed seed for reproducibility

%% Load CSV File
data = readtable('C:\Users\HP\Downloads\nutrient_0.01 (1).csv'); 

% Convert to array
data = table2array(data);

% Extract time and population
time = data(:,1);
population = data(:,2);

%% Add Gaussian Noise
noiseFactor = 0.02; % Adjust this to control noise level
population = population + noiseFactor * randn(size(population)) .* population;

% Ensure no negative values after adding noise
population = max(population, 0);  

% Store original min and max values for later rescaling
popMin = min(population);
popMax = max(population);

% Normalize using Min-Max Scaling
timeNorm = (time - min(time)) ./ (max(time) - min(time));
popNorm = (population - popMin) ./ (popMax - popMin); 

% Split Data (70% Train, 10% Validation, 20% Test)
numSamples = length(timeNorm);
numTrain = floor(0.7 * numSamples);
numVal = floor(0.1 * numSamples);
numTest = numSamples - (numTrain + numVal);

% Ensure Valid Indexing
timeTrain = timeNorm(1:numTrain);
popTrain = popNorm(1:numTrain);
timeVal = timeNorm(numTrain+1:numTrain+numVal);
popVal = popNorm(numTrain+1:numTrain+numVal);
timeTest = timeNorm(numTrain+numVal+1:end);
popTest = popNorm(numTrain+numVal+1:end);

% Convert to cell arrays for LSTM
XTrain = num2cell(timeTrain', 1);
YTrain = num2cell(popTrain', 1);
XVal = num2cell(timeVal', 1);
YVal = num2cell(popVal', 1);
XTest = num2cell(timeTest', 1);
YTest = num2cell(popTest', 1);

%% Train and Validate LSTM Model
layers = [ ...
    sequenceInputLayer(1)
    lstmLayer(64, 'OutputMode', 'sequence')
    batchNormalizationLayer  
    lstmLayer(32, 'OutputMode', 'sequence')
    batchNormalizationLayer  
    fullyConnectedLayer(1)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs', 200, ...
    'MiniBatchSize', 32, ...
    'ValidationData', {XVal, YVal}, ...
    'Shuffle', 'every-epoch', ...
    'L2Regularization', 0.001, ...
    'InitialLearnRate', 0.0005, ...
    'ValidationPatience', 50, ...
    'Plots', 'training-progress', ...
    'Verbose', 0);

% Train the LSTM Model
net = trainNetwork(XTrain, YTrain, layers, options);

% Save the trained model after training
save('trainedLSTM.mat', 'net');

disp("Training and validation completed. Model saved.");

%% Test the Model
load('trainedLSTM.mat', 'net');

% Predict using the trained model
YPred = predict(net, XTest, 'MiniBatchSize', 1);

% Convert cell array predictions to numeric array
YPred = cell2mat(YPred);

% Rescale predictions back to original values
YPred = YPred * (popMax - popMin) + popMin;
popTest = popTest * (popMax - popMin) + popMin;

% Compute RMSE on test data
rmseTest = sqrt(mean((popTest - YPred).^2));
disp(['Final Test RMSE: ', num2str(rmseTest)]);

% Save RMSE with trained model
save('trainedLSTM.mat', 'net', 'rmseTest');

%% Plot Actual vs. Predicted on Test Data
figure;
plot(timeTest, popTest, 'bo', 'LineWidth', 1.5); hold on;
plot(timeTest, YPred, 'r*', 'LineWidth', 1.5);
legend('Actual Population', 'Predicted Population');
title('LSTM Model: Actual vs Predicted (with Noise)');
xlabel('Time');
ylabel('Bacterial Population');
grid on;
