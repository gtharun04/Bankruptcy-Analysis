clc
clear all
close all
warning off
%% Preparing the data
% Importing the data
A=readtable('Smotedata.csv');

% Transforming the data with the required variables for the analysis as
% per our EDA

working_dataset = A(:,[2 39 62 42 86 93 80 68 72]);

%% Splitting the data into two parts and holding the testing part for final testing analysis

cv = cvpartition(size(working_dataset,1),'HoldOut',0.3);
idx = cv.test;
Trainingdata = working_dataset(~idx,:);
Testingdata = working_dataset(idx,:);

%% Preprocessing the traning data
Trainingdata = table2array(Trainingdata); % Converting the Dataset to an array for ease of computation in Matlab

Traininglabels = Trainingdata(:,1); % Defining the Label form the dataset

Traininglabels = cellstr(num2str(Traininglabels));   % Converting to cell array

%% Normalizing the Training data
NormalisedTrainingData = []; % Initializing an empty dataframe

for k=2:size(Trainingdata, 2)
    NormalisedTrainingData = [NormalisedTrainingData, (Trainingdata(:, k) - min(Trainingdata(:, k))) / (max(Trainingdata(:, k)) - min(Trainingdata(:, k)))];
end

%% Using Kfold method to generate indices
indices = crossvalind('Kfold',Traininglabels,10);

%% NB model training and testing on the Trainingdata
nb = classperf(Traininglabels);
for i = 1:10
    test = (indices == i); 
    train = ~test;
    mdlnb = fitcnb(NormalisedTrainingData(train,:),Traininglabels(train));
    [Trainpredictions, scoretrain] = predict(mdlnb,NormalisedTrainingData(test,:));
    classperf(nb,Trainpredictions,test);
end

%% BAM our model is ready, let's test it on the testing data
%% Preprocessing the testing data

Testingdata = table2array(Testingdata); % Converting the Dataset to an array for ease of computation in Matlab

Testinglabels = Testingdata(:,1); % Defining the Label form the dataset

Testinglabels = cellstr(num2str(Testinglabels));   % Converting to cell array

TestingVariables = Testingdata(:,2:end); % Keeping the variables apart

NormalisedTestingData = []; % Initializing an empty dataframe

%% Normalizing the Testing data
for k=1:size(TestingVariables, 2)
    NormalisedTestingData = [NormalisedTestingData, (TestingVariables(:, k) - min(TestingVariables(:, k))) / (max(TestingVariables(:, k)) - min(TestingVariables(:, k)))];
end

%% Testing the performance on the built model

nb2 = classperf(Testinglabels);
[Testpredictions,scoretest] = predict(mdlnb,TestingVariables);
    classperf(nb2,Testpredictions);

%% Testing Output

figure;
confusionchart(Testinglabels,Testpredictions);

[Xnb, Ynb, Tnb,AUCnb] = perfcurve(Testinglabels,scoretest(:,2),'1');

plot(Xnb,Ynb);
legend('nbTrain','nbTest','Location','Best');

idx = (cell2mat(Testinglabels) == '1');
TP = sum(cell2mat(Testinglabels(idx)) == cell2mat(Testpredictions(idx)));
TN = sum(cell2mat(Testinglabels(~idx)) == cell2mat(Testpredictions(~idx)));
p = length(cell2mat(Testinglabels(idx)));
n = length(cell2mat(Testinglabels(~idx)));
FP = n-TP;
FN = p-TN;
N = p+n;

Accuracy = (TP+TN)/(TP+TN+FN+FP);
Recall = TP/(TP+FN);
Precision = TP/(TP+FP);
F1Score = (1+1)*((Precision*Recall)/(1*Precision+Recall));
%% Printing the results

disp('Model               : NB')
fprintf('Training_Accuracy   : %f\n', 1 - nb.ErrorRate);
fprintf('Testing_Accuracy    : %f\n', Accuracy);
fprintf('Recall              : %f\n', Recall);
fprintf('Precision           : %f\n', Precision);
fprintf('F1Score             : %f\n', F1Score);
fprintf('AUC                 : %f\n', AUCnb);
