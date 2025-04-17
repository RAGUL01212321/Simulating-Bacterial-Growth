% Load Data
rng(22);
data = readmatrix('C:\Users\HP\Downloads\nutrient_0.01 (1).csv');
time = data(:,1); % Time column
population = data(:,2); % Population column

% Min-Max Normalization
pop_min = min(population);
pop_max = max(population);
population = (population - pop_min) / (pop_max - pop_min);

% Split Data
numSamples = length(population);
trainSize = floor(0.7 * numSamples);
valSize = floor(0.1 * numSamples);

trainData = population(1:trainSize);
valData = population(trainSize+1:trainSize+valSize);
testData = population(trainSize+valSize+1:end);

% Add Gaussian Noise to Training Data (Mean = 0, Std Dev = 0.01)
noise = 0.04 * randn(size(trainData));
trainDataNoisy = trainData + noise;

% Convert to sequence format (cell array for deep learning input)
trainSeq = num2cell(trainDataNoisy);
valSeq = num2cell(valData);
testSeq = num2cell(testData);

disp('✅ Data Preprocessing Complete');

% Load Data
data = readmatrix('C:\Users\HP\Downloads\nutrient_0.01 (1).csv'); % Update with correct path
time = data(:,1); % Time column
population = data(:,2); % Population column

% Min-Max Normalization
population = (population - min(population)) / (max(population) - min(population));

% Split Data
numSamples = length(population);
trainSize = floor(0.7 * numSamples);
valSize = floor(0.1 * numSamples);

dataTrain = population(1:trainSize);
dataVal = population(trainSize+1:trainSize+valSize);
dataTest = population(trainSize+valSize+1:end);

% Convert to Sequence Format
dataTrainSeq = num2cell(dataTrain);
dataValSeq = num2cell(dataVal);
dataTestSeq = num2cell(dataTest);

% Define GRU Network Architecture
layers = [ 
    sequenceInputLayer(1)
    gruLayer(128, 'OutputMode', 'sequence')  
    dropoutLayer(0.3)  
    gruLayer(64, 'OutputMode', 'sequence')   
    dropoutLayer(0.2) 
    gruLayer(32, 'OutputMode', 'sequence')   
    fullyConnectedLayer(1)
    regressionLayer
];


% Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 180, ...  % Increase epochs
    'MiniBatchSize', 32, ...  % Reduce batch size
    'InitialLearnRate', 0.0001, ...  % Reduce learning rate
    'Shuffle', 'every-epoch', ...
    'ValidationData', {valSeq, valSeq}, ...
    'ValidationFrequency', 15, ...
    'ValidationPatience', 60, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Train the Network
net = trainNetwork(dataTrainSeq, dataTrainSeq, layers, options);

% Save Model
save('GRU_Model.mat', 'net');

% Prediction
predictedSeq = predict(net, dataTestSeq);
predictedValues = cell2mat(predictedSeq);

% Plot Actual vs Predicted
figure;
plot(dataTest, 'b'); hold on;
plot(predictedValues, 'r');
legend('Actual', 'Predicted');
title('GRU Prediction on Test Data');

% Calculate Evaluation Metrics
mse = mean((dataTest - predictedValues).^2);
rmse = sqrt(mse);
mae = mean(abs(dataTest - predictedValues));
ss_total = sum((dataTest - mean(dataTest)).^2);
ss_residual = sum((dataTest - predictedValues).^2);
r2 = 1 - (ss_residual / ss_total);

% Display Results
fprintf('MSE: %.4f\n', mse);
fprintf('RMSE: %.4f\n', rmse);
fprintf('MAE: %.4f\n', mae);
fprintf('R^2 Score: %.4f\n', r2);
