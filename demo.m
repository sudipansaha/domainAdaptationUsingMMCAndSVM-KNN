clc
clear all
close all

load data.mat

%%Treat 50% of data as from source and 50% from target and train
[confusionMatrix,accuracy,predictedLabel]=mmcRetrainKnn(features,labels,floor(0.5*size(features,1)));
confusionMatrix
accuracy