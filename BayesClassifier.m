clc
clear all
close all
load('data.mat');
[x,y,~] = size(face);
image_Mat = zeros(x*y,600);

for i = 1:600
    img = face(:,:,i);
    img_y = reshape(img,x*y,1);                  
    
    image_Mat(:,i) = img_y;
end

%2 Images - Training; 1 Image - Testing
train_data = zeros(x*y,400);
test_data = zeros(x*y,200);

for i=1:200
    ind = randperm(3); %for random selection
%     train_data(:,i*2-1) = image_Mat(:,((i)*3-2)+(ind(1)-1));
%     train_data(:,i*2) = image_Mat(:,((i)*3-2)+(ind(2)-1));
%     test_data(:,i) = image_Mat(:,((i)*3-2)+(ind(3)-1));
    train_data(:,i*2-1) = image_Mat(:,((i)*3-2)+1);
    train_data(:,i*2) = image_Mat(:,((i)*3-2)+ 2);
    test_data(:,i) = image_Mat(:,((i)*3-2));
end

%MLE Estimate for Mean
ml_mean_data = zeros(x*y,200);
for i = 1:200
    ml_mean_data(:,i) = 0.5*(train_data(:,i*2-1)+train_data(:,i*2));
end

%MLE Estimate for Covariance
ml_covariance_data = zeros(x*y,x*y,200);
for i=1:200
    cov_mat = 0.5*( (train_data(:,i*2-1)-ml_mean_data(:,i))*(train_data(:,i*2-1)-ml_mean_data(:,i))' + (train_data(:,i*2)-ml_mean_data(:,i))*(train_data(:,i*2)-ml_mean_data(:,i))' );
    ml_covariance_data(:,:,i) = cov_mat;
end

%Regularize the covariance matrix
pooled_cov = mean(ml_covariance_data,3);
sigma = trace(pooled_cov)/50;

lambda = 0.5;
gamma = 0.5;
shrink_rda_data=zeros(504,504,200);
for i=1:200
    cov_mat = ml_covariance_data(:,:,i);
    shrink_pooled = (1-lambda)*cov_mat + lambda*pooled_cov;
    
    shrink_rda = (1-gamma)*shrink_pooled + gamma*sigma*eye(504,504);
    shrink_rda_data(:,:,i) = shrink_rda;
end

%Classification using the Bayes classifier
acc = 0;
W_i = cell(200,1);
sw_i = cell(200,1);
w_i_o = cell(200,1);

for m=1:200
        cov_mat = shrink_rda_data(:,:,m);
        W_i{m} = -0.5*inv(cov_mat);
        sw_i{m} = inv(cov_mat)*ml_mean_data(:,m);
        w_i_o{m} = -0.5*ml_mean_data(:,m)'*inv(cov_mat)*ml_mean_data(:,m) - 0.5*log(det(cov_mat)) + log(1/200);    
       
end

for l = 1:200
    ftest = test_data(:,l);
    dmat = zeros(200,1);
    
    for m=1:200
        dmat(m) = ftest'*W_i{m}*ftest + sw_i{m}'*ftest + w_i_o{m};
    end
    
    [val, max_id] = max(dmat);
    
    if max_id == l
       acc = acc+1;
    end
    
end

accuracy = (acc/200)*100;



   
        
    


