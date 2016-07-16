function [mappedX, mapping] = pcaP1(X, no_dims)
%pcaP1 Performs the PCA algorithm
%
%   [mappedX, mapping] = pcaP1(X, no_dims)
%
% The function runs PCA on a set of datapoints X. The variable
% no_dims sets the number of dimensions of the feature points in the 
% embedded feature space (no_dims >= 1, default = 2). 
% The function returns the locations of the embedded trainingdata in 
% mappedX. Furthermore, it returns information on the mapping in mapping.

    if ~exist('no_dims', 'var')
        no_dims = 2;
    end
	
	% Make sure data is zero mean
    mapping.mean = mean(X, 1);
	X = bsxfun(@minus, X, mapping.mean);

	% Compute covariance matrix
    if size(X, 2) < size(X, 1)
        C = cov(X);
    else
        C = (1 / size(X, 1)) * (X * X');        % if N>D, we better use this matrix for the eigendecomposition
    end
	
	% Perform eigendecomposition of C
	C(isnan(C)) = 0;
	C(isinf(C)) = 0;
    [M, lambda] = eig(C);
    
    % Sort eigenvectors in descending order
    [lambda, ind] = sort(diag(lambda), 'descend');
    if no_dims < 1
        no_dims = find(cumsum(lambda ./ sum(lambda)) >= no_dims, 1, 'first');
        disp(['Embedding into ' num2str(no_dims) ' dimensions.']);
    end
    if no_dims > size(M, 2)
        no_dims = size(M, 2);
        warning(['Target dimensionality reduced to ' num2str(no_dims) '.']);
    end
	M = M(:,ind(1:no_dims));
    lambda = lambda(1:no_dims);
	
	% Apply mapping on the data
    if ~(size(X, 2) < size(X, 1))
        M = bsxfun(@times, X' * M, (1 ./ sqrt(size(X, 1) .* lambda))');     % normalize in order to get eigenvectors of covariance matrix
    end
    mappedX = X * M;
    
    % Store information for out-of-sample extension
    mapping.M = M;
	mapping.lambda = lambda;
    