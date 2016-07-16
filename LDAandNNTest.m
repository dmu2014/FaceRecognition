clc
clear all
close all

load('data.mat');

FV = zeros(600, 504);
LV = zeros(600,1);
n=1;
for j = 1:1:600
    FV(j,:) = reshape(face(:,:,j), [1, 504]);
    if mod(j,3) ~= 0
        LV(j,1) = floor(j/3)+1;
    else
        LV(j,1) = j/3;
        
    end
end

[mappedX, mapping] = lda(FV, LV, 199);
FVred = mappedX;
misclassification = 0;
for i = 1 : 1 : size(face,3)
        if mod(i,3) == 0
%             ftest1 = reshape(face(:,:,i), [1, (size(face,1)*size(face,2))]);
%             ftest = ftest1 * mapping.M;
            ftest = FVred(i,:);
            result = knnclassification(ftest, FVred ,LV, 5, '2norm');
            if result ~= i/3
                misclassification = misclassification + 1 ;
            end
        end
    
end
NNaccuracy = 1 - misclassification/200