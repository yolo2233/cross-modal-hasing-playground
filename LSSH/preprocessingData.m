function X = preprocessingData(X,P,meanX) 

%
% Reference:
% Jile Zhou, GG Ding, Yuchen Guo
% "Latent Semantic Sparse Hashing for Cross-modal Similarity Search"
% ACM SIGIR 2014
% (Manuscript)
%
% Version1.0 -- Nov/2013
% Written by Jile Zhou (zhoujile539@gmail.com)
%

%centering
X = bsxfun(@minus,X, meanX);
%embedding
X = P'* X;
norm_y = sum(X.^2,1).^-.5;
for i = 1 : size(X,1)
    X(i,:) = X(i,:).*norm_y;
end
