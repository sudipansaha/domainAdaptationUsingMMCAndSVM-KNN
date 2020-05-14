function [confusionMatrix,accuracy,predictedLabel]=mmcRetrainKnn(inputData,inputDataTrueLabel,trainingTestDemarcation,trainingTestOrder,clusterNumber)

%% Defining input and output variables

%%Input: inputData- Input Data
%%True label of input data - inputDataTrueLabel
%%trainingTestDemarcation- Data point where training (test) data ends &
%%test (training) data starts
%%trainingTestOrder- if 1, data before trainingTestDemarcation is set to be training, if 0 opposite
%%clusterNumber - the number of clusters

%%Output
%%confusionMatrix - confusion Matrix
%%accuracy - accuracy
%%predictedLabel- final label of max-margin clustering

clc

%% Assigning values to optional arguments
if nargin<5
    clusterNumber=max(inputDataTrueLabel);
end
if nargin<4
    trainingTestOrder=1;
end
if nargin<3
    trainingTestDemarcation=floor(0.6*size(inputData,1));
end



%% Applying K-Means clustering & initializing resMatrix, probMatrix

kmeansCluster=kmeans(inputData,clusterNumber);  %%Applying kmeans clustering to the input data
resMatrix=zeros(size(inputData,1),max(kmeansCluster)); %%initializing resultMatrix
probMatrix=zeros(size(inputData,1),max(kmeansCluster)); %%initializing probability Matrix




%% Applying Maximum Margin clustering and obtaining finalLaebl
for iterKMeansClusters=1:max(kmeansCluster)
    tempLabels=-ones(size(inputData,1),1);
    tempLabels(find(kmeansCluster==iterKMeansClusters))=1; %#ok<FNDSB>
    [predictedLabels,~,probabilityScore] = iterativeSVR(inputData, 0.05, 1, tempLabels, 0.2, 0.3);
    resMatrix(:,iterKMeansClusters)=predictedLabels;
    for iterOverProbScore=1:size(probabilityScore,1)
        probMatrix(iterOverProbScore,iterKMeansClusters)=1/(1+exp(-probabilityScore(iterOverProbScore)));
    end
end

outputLabel=zeros(size(inputData,1),1);

for iterOutputLabel=1:size(resMatrix,1)
    temp=probMatrix(iterOutputLabel,:);
    [~,indexMax]=max(temp);
    outputLabel(iterOutputLabel,1)=indexMax;
end



%% Finding low confidence points, for kNN
probMatrixReordered=zeros(size(probMatrix));
for probMatrixReorderingIter=1:size(probMatrixReordered,1)
    probMatrixReordered(probMatrixReorderingIter,:)=sort (probMatrix(probMatrixReorderingIter,:),'descend');
end
confidenceScore=probMatrixReordered(:,1)-probMatrixReordered(:,2);
[~,sortLowConfidenceIndex]=sort(confidenceScore);
selectedLowConfidenceIndex=sortLowConfidenceIndex(1:floor(0.2*length(sortLowConfidenceIndex)));
lowConfidenceIpData=inputData(selectedLowConfidenceIndex,:);

knnMatrixForLowConfidenceIpData=knnsearch(inputData,lowConfidenceIpData,'K',10,'distance','seuclidean');
knnMatrixForLowConfidenceIpData=knnMatrixForLowConfidenceIpData(:,2:10); %%1st point will be data itself




for knnSvmIter=1:size(lowConfidenceIpData,1)
    toPredictData=lowConfidenceIpData(knnSvmIter,:);
    toPredictDataInitialGuess=outputLabel(selectedLowConfidenceIndex(knnSvmIter));
    trainingDataIndexes=knnMatrixForLowConfidenceIpData(knnSvmIter,:);
    trainingData=inputData(trainingDataIndexes,:);
    trainingDataLabels=outputLabel(trainingDataIndexes);
    
    if std(trainingDataLabels)==0
        outputLabel(selectedLowConfidenceIndex(knnSvmIter))=trainingDataLabels(1);
    else
        
        model=svmtrain2(trainingDataLabels,trainingData,'-t 4 -c 2 -b 1');
        [prediction, ~, ~]=svmpredict(toPredictDataInitialGuess, toPredictData, model,'-b 1');
        outputLabel(selectedLowConfidenceIndex(knnSvmIter))=prediction;
    end
    
end


%% Mapping from obtained labels of outputLabel to inputDataTrueLabel
if trainingTestOrder==1
    trueLabelForTrainingData=inputDataTrueLabel(1:trainingTestDemarcation);
    outputLabelForTrainingData=outputLabel(1:trainingTestDemarcation);
else
    trueLabelForTrainingData=inputDataTrueLabel(trainingTestDemarcation+1:end);
    outputLabelForTrainingData=outputLabel(trainingTestDemarcation+1:end);
end


trueLabelCorrespondingToPredictedLabel=zeros(1,max(outputLabelForTrainingData));
for iterLabelMatching=1:max(outputLabelForTrainingData)
    correspondingTrueData=trueLabelForTrainingData(find(outputLabelForTrainingData==iterLabelMatching)); %#ok<FNDSB>
    trueLabelCorrespondingToPredictedLabel(iterLabelMatching)=mode(correspondingTrueData);
end

outputLabelMatrix=zeros(length(outputLabel),max(outputLabelForTrainingData));
for iterLabelAssignment=1:max(outputLabelForTrainingData)
    outputLabelMatrix(:,iterLabelAssignment)=(outputLabel==iterLabelAssignment);
    
end

predictedLabel=zeros(size(outputLabel));
for iterTrueLabelAssignment=1:max(outputLabelForTrainingData)
    predictedLabel(find(outputLabelMatrix(:,iterTrueLabelAssignment)))=trueLabelCorrespondingToPredictedLabel(iterTrueLabelAssignment); %#ok<FNDSB>
end

%% Making confusion matrix
if trainingTestOrder==1
    trueLabelForTestData=inputDataTrueLabel(trainingTestDemarcation+1:end);
    predictedLabelForTestData=predictedLabel(trainingTestDemarcation+1:end);
else
    trueLabelForTestData=inputDataTrueLabel(1:trainingTestDemarcation);
    predictedLabelForTestData=predictedLabel(1:trainingTestDemarcation);
end

confusionMatrix=confusionmat(trueLabelForTestData,predictedLabelForTestData);
accuracy=length(find(trueLabelForTestData==predictedLabelForTestData))/length(trueLabelForTestData);






