% iterative SVR algorithm for binary clustering

% dataMatrix:      N by D data matrix
% sigma:  RBF kernel width K(x)=exp(|x|^2/sigma^2)
% regularizationParam:      regularization parameter
% initialLabels:      initial labels for iterSVR to start
%         (usually chosen as k-means clusterign result)
% loss:   epsilon-sensitive loss in SVR
%         theoretically, the smaller, the better
%         in practice may be chosen in domain [0,0.2]
% balanceParameter:    balance parameter, a ratio in (0, 0.5)


function [finalLabels, model,pdt] = iterativeSVR(dataMatrix, sigma, regularizationParam, initialLabels, loss, balanceParameter)
addpath('./svmFunctions/');

[sizeData, ~] = size(dataMatrix);
gamma = 1/sigma^2;

st= zeros(1,50);
for ite = 1:50
    
    opt = sprintf('-s 3 -t 4 -g %g -c %g -p %g -e 0.001 -b 1', gamma, regularizationParam, loss);
    model = svmtrain2(initialLabels, dataMatrix, opt);
    %model
    [pdt, ~, ~] = svmpredict(initialLabels, dataMatrix, model);
    %pdt
    
    bias = calBiasSVR(pdt, sizeData, balanceParameter);
    pdt = pdt - bias;
    
    finalLabels=initialLabels;
    finalLabels((pdt>=0)) = 1;
    finalLabels((pdt<0)) =-1;
    st(ite) = norm(finalLabels - pdt);
    if(ite>1 && abs( st(ite) - st(ite-1)) <= 1e-3 * st(ite - 1))
        break
    end
end