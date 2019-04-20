function [X1_train, X1_binary, X2_train, X2_binary] = crossMediaTraining(X1, X2, n2, beta, lambda, L1, L2, X1_test, X2_test,r)

%% views contains all the training data from different views.
% X1 and X2 should be m1*(n1+n2), m2*(n2+n3), should be centralized
% where n2 is the number of images with tags
% beta and lambda are the paras
% L1 and L2 are the laplacian matrix for images and texts
%% Prepare the data
beta=10^beta;
lambda=10^lambda;
[m1, imgNum] = size(X1);
[m2, textNum] = size(X2);
n1 = imgNum-n2;
n3 = textNum-n2;
tmp1 =  diag(ones(1,n1+n2));
S1 = tmp1(n1+1:n1+n2,:);clear tmp1;
tmp2 = diag(ones(1,n2+n3));
S2 = tmp2(1:n2,:);clear tmp2;
U = 10000*diag(ones(1,n2));

% oneline = ones(trainNum,1);
eyemat_1 = eye(imgNum);
eyemat_2 = eye(textNum);
eyemat_m_1 = eye(m1);
eyemat_m_2 = eye(m2);
M1 = X1*X1' + beta*eyemat_m_1;
M2 = X2*X2' + beta*eyemat_m_2;
A1 = eyemat_1 - (X1')/M1*X1; %B in the paper
M2_m = (M2)^(-1);
A2 = eyemat_2 - (X2')*M2_m*X2;


%% Get Laplacian Matrix L1 and L2
% para.lamda = 1;
% para.k =5;
% [L1] = Laplacian_GK(X1, para);
% [L2] = Laplacian_GK(X2, para);
% save Laplacian_GK L1 L2;
% load Laplacian_GK;

%% Get Y1 (v, eigval)

% D = lambda*eyemat-lambda*lambda*eyemat/(L1+lambda*eyemat) + lambda*eyemat-lambda*lambda*eyemat/(L2+lambda*eyemat) + alpha*(Lc-Lc*X'/(X*Lc*X'+beta*eyemat_m)*X*Lc);
C2 = (A2+lambda*L2+S2'*U*S2)\S2'*U*S2;%E in the paper
D = A1+C2'*A2*C2 + (S1-S2*C2)'*U*(S1-S2*C2) + lambda*L1+lambda*C2'*L2*C2;%C in the paper
% clear L1 L2;
D=(D+D')/2;
[v,eigval]=eig(D);
eigval = diag(eigval);

[eigval, idx] = sort(eigval);
Y1 = v(:,idx(2:r+1));%F in the paper
Y2 = C2*Y1;
% [eigval, idx] = sort(diag(eigval));
% topk=100;
% Y = v(idx(1:topk),:);
% Y = Y';
% % save Y Y;
% W = (X*Lc*X'+beta*eyemat_m)\X*Lc*Y;
% % save W W;
% b = (oneline'*Y-oneline'*X'*W)/trainNum;
% % save b b;


% codeLen=8:16:144;
% steps_num=length(codeLen);
% % maps=zeros(steps_num,24);
% maps=zeros(2,10);
% for len=3:10
    % Get Y, W, b
    
    eyemat_m1 = eye(m1);
    eyemat_m2 = eye(m2);
    W1 = (X1*X1'+beta*eyemat_m1)\X1*Y1;
    W2 = (X2*X2'+beta*eyemat_m2)\X2*Y2;

    % Start to query
    [feaDimX, vid_num] = size(X1_test');
%     oneline = ones(vid_num,1);
    X1_lowD = X1_test*W1;   %low dimensity kf
    %clear W1;
    X1_med = median(X1_lowD);
    X1_binary=(X1_lowD>repmat(X1_med,vid_num,1));
    
    X2_lowD = X2_test*W2;   %low dimensity kf
    %clear W2;
    X2_med = median(X2_lowD);
    X2_binary=(X2_lowD>repmat(X2_med,vid_num,1));
    
    X1 = X1';
    X2 = X2';
    [feaDimX, vid_num] = size(X1');

    X1_lowD = X1*W1;   %low dimensity kf
    %clear W1;
    X1_med = median(X1_lowD);
    X1_train=(X1_lowD>repmat(X1_med,vid_num,1));
    
    X2_lowD = X2*W2;   %low dimensity kf
    %clear W2;
    X2_med = median(X2_lowD);
    X2_train=(X2_lowD>repmat(X2_med,vid_num,1));
    
%     filev1=['result\Y1_wiki_low',num2str(n2),num2str(beta+6),num2str(lambda+6)];
%     filee1=['result\Y1_wiki_bin',num2str(n2),num2str(beta+6),num2str(lambda+6)];
%     filev2=['result\Y2_wiki_low',num2str(n2),num2str(beta+6),num2str(lambda+6)];
%     filee2=['result\Y2_wiki_bin',num2str(n2),num2str(beta+6),num2str(lambda+6)];
%     
%     save (filev1,'X1_lowD','-ascii');
%     save (filee1,'X1_binary','-ascii');
%     save (filev2,'X2_lowD','-ascii');
%     save (filee2,'X2_binary','-ascii');








