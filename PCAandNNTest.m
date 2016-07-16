clc
clear all
close all

load('data.mat');

FV = zeros(400, 504);
LV = zeros(400,1);
n=1;
for j = 1:1:600
    if mod(j,3) ~= 0
        FV(n,:) = reshape(face(:,:,j), [1, 504]);
        LV(n,1) = floor(j/3)+1;
        n=n+1;
        
    end
end

[mappedX, mapping] = pcaP1(FV, 20);
FVred = mappedX;
misclassification = 0;
for i = 1 : 1 : size(face,3)
        if mod(i,3) == 0
            ftest1 = reshape(face(:,:,i), [1, (size(face,1)*size(face,2))]);
            ftest = ftest1 * mapping.M;
            result = knnclassification(ftest, FVred ,LV, 3, '2norm');
            if result ~= i/3
                misclassification = misclassification + 1 ;
            end
        end
    
end
NNaccuracy = 1 - misclassification/200