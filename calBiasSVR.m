function calVal = calBiasSVR(pdt, sizeData, balanceParameter)

halfSizeData = floor(sizeData/2);
balancedSize = floor(sizeData*balanceParameter);
Y = zeros(sizeData,1);

c = halfSizeData - balancedSize - 1;
err = zeros(2*balancedSize+1,1);
B = zeros(2*balancedSize+1,1);
[sortedPdt, ~] = sort(pdt, 'ascend');
for iter = (halfSizeData-balancedSize):(halfSizeData+balancedSize)
    Y(1:iter) = -1;
    Y(iter+1:sizeData) = 1;
    b = (sortedPdt(iter) + sortedPdt(iter+1))/2;
    xi = abs(Y - sortedPdt + b);
    err(iter - c) = sum(xi);
    B(iter-c) = b;
end
dex = find(err == min(err));
calVal = B(dex(1));