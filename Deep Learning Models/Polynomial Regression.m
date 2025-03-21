% MATLAB Code for Bacterial Growth Prediction using Polynomial Regression

% Step 1: Load the Data from CSV
filePath = 'C:\Users\HP\Downloads\nutrient_0.01 (1).csv'; 
data = readtable(filePath, 'PreserveVariableNames', true);
data.Properties.VariableNames = strtrim(data.Properties.VariableNames);

% Extract Time and Population from the table
time = data.Time; % Column named 'Time'
population = data.Population; % Column named 'Population'

% Step 2: Normalize the Data
timeMin = min(time);
timeMax = max(time);
timeNorm = (time - timeMin) / (timeMax - timeMin);

popMin = min(population);
popMax = max(population);
popNorm = (population - popMin) / (popMax - popMin);

% Split into Training and Testing Sets (80% train, 20% test)
trainRatio = 0.8;
numTrain = floor(trainRatio * length(time));

timeTrain = timeNorm(1:numTrain);
popTrain = popNorm(1:numTrain);
timeTest = timeNorm(numTrain+1:end);
popTest = popNorm(numTrain+1:end);

% Step 3: Train Polynomial Regression Model
degree = 6; % Polynomial degree
numEpochs = 6500; % Training iterations
initialLearningRate = 0.01; % Initial learning rate
learningRateDecay = 0.999; % Decay for learning rate
lambda = 0.001; % Regularization strength 

% Initialize polynomial coefficients randomly 
polyCoeffs = 0.1 * rand(degree+1, 1); 

% Create a design matrix for polynomial regression
XTrain = zeros(numTrain, degree+1);
for d = 0:degree
    XTrain(:, d+1) = timeTrain.^d;
end

% Track training loss over epochs
trainingLoss = zeros(numEpochs, 1);

% Gradient descent optimization with Ridge Regularization
for epoch = 1:numEpochs
    % Compute predictions
    popPredTrain = XTrain * polyCoeffs;
    
    % Compute training loss (Mean Squared Error)
    error = popPredTrain - popTrain;
    trainingLoss(epoch) = mean(error.^2);
    
    % Compute gradient with Ridge Regularization (L2)
    gradient = (2/numTrain) * (XTrain' * error) + lambda * polyCoeffs; 
    
    % Adjust learning rate with decay
    currentLearningRate = initialLearningRate * (learningRateDecay ^ epoch);
    
    % Update coefficients using gradient descent
    polyCoeffs = polyCoeffs - currentLearningRate * gradient;
    
    % Limit coefficient range to avoid instability
    polyCoeffs = max(min(polyCoeffs, 50), -50);
end

% Step 4: Make Predictions on Training and Testing Data
popPredTrain = XTrain * polyCoeffs;

% Testing predictions
XTest = zeros(length(timeTest), degree+1);
for d = 0:degree
    XTest(:, d+1) = timeTest.^d;
end
popPredTest = XTest * polyCoeffs;

% Denormalize predictions
popPredTrain = popPredTrain * (popMax - popMin) + popMin;
popPredTest = popPredTest * (popMax - popMin) + popMin;
popTrainDenorm = popTrain * (popMax - popMin) + popMin;
popTestDenorm = popTest * (popMax - popMin) + popMin;

% Step 5: Calculate RMSE (Root Mean Square Error)
rmseTrain = sqrt(mean((popPredTrain - popTrainDenorm).^2));
rmseTest = sqrt(mean((popPredTest - popTestDenorm).^2));
fprintf('Training RMSE: %.4f\n', rmseTrain);
fprintf('Testing RMSE: %.4f\n', rmseTest);

% Step 6: Predict Future Values (Next 5 Time Steps)
futureSteps = 5;
futureTimeNorm = (time(end) + (1:futureSteps) * (time(2) - time(1)) - timeMin) / (timeMax - timeMin);
XFuture = zeros(futureSteps, degree+1);
for d = 0:degree
    XFuture(:, d+1) = futureTimeNorm.^d;
end
futurePredictionsNorm = XFuture * polyCoeffs;
futurePredictions = futurePredictionsNorm * (popMax - popMin) + popMin;
futureTime = time(end) + (1:futureSteps) * (time(2) - time(1));

% Step 7: Plot Actual vs Predicted Values
figure;
plot(time(1:numTrain), population(1:numTrain), 'b-', 'LineWidth', 2, 'DisplayName', 'Actual (Training)');
hold on;
plot(time(numTrain+1:end), population(numTrain+1:end), 'b--', 'LineWidth', 2, 'DisplayName', 'Actual (Testing)');
plot(time(1:numTrain), popPredTrain, 'r-', 'LineWidth', 2, 'DisplayName', 'Predicted (Training)');
plot(time(numTrain+1:end), popPredTest, 'r--', 'LineWidth', 2, 'DisplayName', 'Predicted (Testing)');
plot(futureTime, futurePredictions, 'g-', 'LineWidth', 2, 'DisplayName', 'Future Predictions');
xlabel('Time');
ylabel('Population');
title('Actual vs Predicted Bacterial Population Growth (Polynomial Regression with Ridge)');
legend('show');
grid on;

% Step 8: Plot Learning Curve (Epochs vs Training Loss)
figure;
semilogy(1:numEpochs, trainingLoss, 'b-', 'LineWidth', 2, 'DisplayName', 'Training Loss');
xlabel('Epoch');
ylabel('Training Loss (MSE)');
title('Learning Curve (Epochs vs Training Loss)');
legend('show');
grid on;
