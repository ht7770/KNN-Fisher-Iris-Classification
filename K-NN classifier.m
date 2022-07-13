close all; 
clear all; 

load fisheriris
X = meas;
Y = species;
knnTrainErrorCount = 0;

knnTestErrorCount = 0;
knnTrainingErrorAVG = 0;
knnTestingErrorAVG = 0;
trainingX = [];
testingX = [];

% SHUFFLING DATA %
newRowOrder = randperm(150);
Xshuffled = X(newRowOrder, :);
Yshuffled = Y(newRowOrder, :);

% X SPLIT FOR CROSS VALIDATION %
x1 = Xshuffled(1:30, :);
x2 = Xshuffled(31:60, :);
x3 = Xshuffled(61:90, :);
x4 = Xshuffled(91:120, :);
x5 = Xshuffled(121:150, :);
xTrain5 = [x1;x2;x3;x4];
xTrain4 = [x1;x2;x3;x5];
xTrain3 = [x1;x2;x4;x5];
xTrain2 = [x1;x3;x4;x5];
xTrain1 = [x2;x3;x4;x5];

% Y SPLIT FOR CROSS VALIDATION %
y1 = Yshuffled(1:30, :);
y2 = Yshuffled(31:60, :);
y3 = Yshuffled(61:90, :);
y4 = Yshuffled(91:120, :);
y5 = Yshuffled(121:150, :);
yTrain5 = [y1;y2;y3;y4];
yTrain4 = [y1;y2;y3;y5];
yTrain3 = [y1;y2;y4;y5];
yTrain2 = [y1;y3;y4;y5];
yTrain1 = [y2;y3;y4;y5];


% CALCULATING THE AVG TRAINING AND TESTING ERROR WHERE K IS 1 - 5 %
disp('****** K-NEAREST NEIGHBOURS ******')
for k = 1 : 10

    % ***** FOLD 1 ***** %
    knnTRAIN = fitcknn(xTrain1, yTrain1, 'NumNeighbors', k);
    % prediction using training data %
    knnTrainPrediction = predict(knnTRAIN, xTrain1);
    % prediction using unseen testing data %
    knnTestPrediction = predict(knnTRAIN, x1);

    % calculating training error for this fold %
    for i = 1 : length(knnTrainPrediction)
        if strcmp(knnTrainPrediction(i), yTrain1(i)) == 0
            knnTrainErrorCount = knnTrainErrorCount + 1;
        end
    end
    knnTrainingError = (knnTrainErrorCount / 120) * 100;
    knnTrainingErrorAVG = knnTrainingErrorAVG + knnTrainingError;

    % Calculating testing error for this fold %
    for i = 1 : length(knnTestPrediction)
        if strcmp(knnTestPrediction(i), y1(i)) == 0
            knnTestErrorCount = knnTestErrorCount + 1;
        end
    end
    knnTestingError = (knnTestErrorCount / 30) * 100;
    knnTestingErrorAVG = knnTestingErrorAVG + knnTestingError;

    % ***** FOLD 2 ***** %
    knnTRAIN = fitcknn(xTrain2, yTrain2, 'NumNeighbors', k);
    % prediction using training data %
    knnTrainPrediction = predict(knnTRAIN, xTrain2);
    % prediction using unseen testing data %
    knnTestPrediction = predict(knnTRAIN, x2);

    % calculating training error for this fold %
    for i = 1 : length(knnTrainPrediction)
        if strcmp(knnTrainPrediction(i), yTrain2(i)) == 0
            knnTrainErrorCount = knnTrainErrorCount + 1;
        end
    end
    knnTrainingError = (knnTrainErrorCount / 120) * 100;
    knnTrainingErrorAVG = knnTrainingErrorAVG + knnTrainingError;

    % Calculating testing error for this fold %
    for i = 1 : length(knnTestPrediction)
        if strcmp(knnTestPrediction(i), y2(i)) == 0
            knnTestErrorCount = knnTestErrorCount + 1;
        end
    end
    knnTestingError = (knnTestErrorCount / 30) * 100;
    knnTestingErrorAVG = knnTestingErrorAVG + knnTestingError;

    % ***** FOLD 3 ***** %
    knnTRAIN = fitcknn(xTrain3, yTrain3, 'NumNeighbors', k);
    % prediction using training data %
    knnTrainPrediction = predict(knnTRAIN, xTrain3);
    % prediction using unseen testing data %
    knnTestPrediction = predict(knnTRAIN, x3);
    % calculating training error for this fold %
    for i = 1 : length(knnTrainPrediction)
        if strcmp(knnTrainPrediction(i), yTrain3(i)) == 0
            knnTrainErrorCount = knnTrainErrorCount + 1;
        end
    end
    knnTrainingError = (knnTrainErrorCount / 120) * 100;
    knnTrainingErrorAVG = knnTrainingErrorAVG + knnTrainingError;
    % Calculating testing error for this fold %
    for i = 1 : length(knnTestPrediction)
        if strcmp(knnTestPrediction(i), y3(i)) == 0
            knnTestErrorCount = knnTestErrorCount + 1;
        end
    end
    knnTestingError = (knnTestErrorCount / 30) * 100;
    knnTestingErrorAVG = knnTestingErrorAVG + knnTestingError;

    % ***** FOLD 4 ***** %
    knnTRAIN = fitcknn(xTrain4, yTrain4, 'NumNeighbors', k);
    % prediction using training data %
    knnTrainPrediction = predict(knnTRAIN, xTrain4);
    % prediction using unseen testing data %
    knnTestPrediction = predict(knnTRAIN, x4);

    % calculating training error for this fold %
    for i = 1 : length(knnTrainPrediction)
        if strcmp(knnTrainPrediction(i), yTrain4(i)) == 0
            knnTrainErrorCount = knnTrainErrorCount + 1;
        end
    end
    knnTrainingError = (knnTrainErrorCount / 120) * 100;
    knnTrainingErrorAVG = knnTrainingErrorAVG + knnTrainingError;

    % Calculating testing error for this fold %
    for i = 1 : length(knnTestPrediction)
        if strcmp(knnTestPrediction(i), y4(i)) == 0
            knnTestErrorCount = knnTestErrorCount + 1;
        end
    end
    knnTestingError = (knnTestErrorCount / 30) * 100;
    knnTestingErrorAVG = knnTestingErrorAVG + knnTestingError;

    % ***** FOLD 5 ***** %
    knnTRAIN = fitcknn(xTrain5, yTrain5, 'NumNeighbors', k);
    % prediction using training data %
    knnTrainPrediction = predict(knnTRAIN, xTrain5);
    % prediction using unseen testing data %
    knnTestPrediction = predict(knnTRAIN, x5);

    % calculating training error for this fold %
    for i = 1 : length(knnTrainPrediction)
        if strcmp(knnTrainPrediction(i), yTrain5(i)) == 0
            knnTrainErrorCount = knnTrainErrorCount + 1;
        end
    end
    knnTrainingError = (knnTrainErrorCount / 120) * 100;
    knnTrainingErrorAVG = knnTrainingErrorAVG + knnTrainingError;
    % Calculating testing error for this fold %
    for i = 1 : length(knnTestPrediction)
        if strcmp(knnTestPrediction(i), y5(i)) == 0
            knnTestErrorCount = knnTestErrorCount + 1;
        end
    end
    knnTestingError = (knnTestErrorCount / 30) * 100;
    knnTestingErrorAVG = knnTestingErrorAVG + knnTestingError;

    output = [' FOR K at: ', num2str(k)];
    disp(output)
    knnTrainingErrorAVG = knnTrainingErrorAVG / 5
    knnTestingErrorAVG = knnTestingErrorAVG / 5
    trainingX = [trainingX knnTrainingErrorAVG];
    testingX = [testingX knnTestingErrorAVG];
    knnTrainErrorCount = 0;
    knnTestErrorCount = 0;
end 

Ky = 1:10;

% plotting training error over different values of K %
ax1 = nexttile;
plot(ax1, Ky, trainingX)
title(ax1, 'Training Error at different Ks');
ylabel(ax1, '% Training Error');
xlabel(ax1, 'K-Value');

% plotting testing error over different values of K %
ax2 = nexttile;
plot(ax2, Ky, testingX)
title(ax2, 'Testing Error at Different Ks');
ylabel(ax2, '% Training Error');
xlabel(ax2, 'K-Value');

