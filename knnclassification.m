function label = knnclassification(testsamplesX,samplesX, samplesY, Knn,type)

% Classify using the Nearest neighbor algorithm
% Inputs:
% 	samplesX	   - Train samples
%	samplesY	   - Train labels
%   testsamplesX   - Test  samples
%	Knn		       - Number of nearest neighbors 
%
% Outputs
%	label	- Predicted targets

if nargin < 5
    type = '2norm';
end

L			= length(samplesY);
Uc          = unique(samplesY);

if (L < Knn),
   error('More neighbors than Points')
end

N                   = size(testsamplesX, 1);
label              = zeros(N,1); 
switch type
case '2norm'
    for i = 1:N,
        dist            = sum((samplesX - ones(L,1)*testsamplesX(i,:)).^2,2);
        [m, indices]    = sort(dist);  
        n               = hist(samplesY(indices(1:Knn)), Uc);
        [m, best]       = max(n);
        label(i)        = Uc(best);
    end
case '1norm'
    for i = 1:N,
        dist            = sum(abs(samplesX - ones(L,1)*testsamplesX(i,:)),2);
        [m, indices]    = sort(dist);   
        n               = hist(samplesY(indices(1:Knn)), Uc);
        [m, best]       = max(n);
        label(i)        = Uc(best);
    end

otherwise
    error('Unknown measure function');
end
