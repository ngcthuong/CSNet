function net = CSNet_init
global featureSize noLayer blkSize subRate isLearnMtx;

test = 0;
if test == 1
    featureSize = 64;
    noLayer = 7; 
    blkSize = 32; 
    subRate = 0.1; 
end

noMeas = round(subRate * blkSize ^2); 

%%% 17 layers
b_min = 0.025;
lr11  = [1 1];
lr10  = [1 0];
lr00  = [0 0];
weightDecay = [1 0];
meanvar  =  [zeros(featureSize,1,'single'), 0.01*ones(featureSize,1,'single')];

% Define network
net.layers = {} ;

%% 1. Sampling layer - for gray image 
% Sampling network, with kernel size of blkSize x blkSize, do no use
% bias --> initialized as zero and learn rate = 0. 

% Load sensing matrix of size blkSizexBlkSize 
trial = 1; 
fileName = ['SensingMtxs\BlkSize' num2str(blkSize) '_trial' num2str(trial) '.mat' ];
if ~(exist(fileName))
    Phi_Full = orth(rand(blkSize^2, blkSize^2));
    save(fileName, 'Phi_Full'); 
else
    load(fileName); 
    Phi = single(Phi_Full(1:noMeas, :)); 
end

net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{zeros(blkSize, blkSize, 1, noMeas,'single'), zeros(noMeas,1,'single')}}, ...
    'stride', blkSize, ...
    'pad', 0, ...
    'dilate',1, ...
    'learningRate', isLearnMtx, ...
    'weightDecay',weightDecay, ...
    'opts',{{}}) ;
% net.layers{end+1} = struct('type', 'relu','leak',0) ; -- do not use relu 

% assign the sampling matrix
W = zeros(blkSize, blkSize, 1, noMeas); 
for i = 1:1:noMeas
    W(:, :, 1, i) = reshape(Phi(i, :), blkSize, blkSize); 
end
net.layers{1}.weights(1) = {single(W)}; 

% im = double(imread('cameraman.tif'));

%% 2. Initial reconstruction layer with 1x1 Convolution 
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{zeros(1, 1, noMeas, blkSize*blkSize,'single'), zeros(blkSize*blkSize,1,'single')}}, ...
    'stride', 1, ...
    'pad', 0, ...
    'dilate',1, ...
    'learningRate',lr10, ...
    'weightDecay',weightDecay, ...
    'opts',{{}}) ;

W2 = zeros(1, 1, noMeas, blkSize*blkSize); 
PhiInv = pinv(Phi); 
for i = 1:1:noMeas
    W2(:, :, i, :) = PhiInv(:, i); 
end
net.layers{2}.weights(1) = {single(W2)}; 

%% 3. Reshape and concatinate to make recon. image 
net.layers{end+1} = struct('type', 'reshapeconcat');

%% 4. Reconstruction network - DnCNN 
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{sqrt(2/(9*featureSize))*randn(3,3,1,featureSize,'single'), zeros(featureSize,1,'single')}}, ...
    'stride', 1, ...
    'pad', 1, ...
    'dilate',1, ...
    'learningRate',[1 0.5], ... 
    'weightDecay',weightDecay, ...
    'opts',{{}}) ;
net.layers{end+1} = struct('type', 'relu','leak',0) ;

for i = 1:1:noLayer - 2
    
    net.layers{end+1} = struct('type', 'conv', ...
        'weights', {{sqrt(2/(9*featureSize))*randn(3,3,featureSize,featureSize,'single'), zeros(featureSize,1,'single')}}, ...
        'stride', 1, ...
        'learningRate',[1 0.5], ...
        'dilate',1, ...
        'weightDecay',weightDecay, ...
        'pad', 1, 'opts', {{}}) ;
    
    net.layers{end+1} = struct('type', 'relu','leak',0) ;
    
end

net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{sqrt(2/(9*featureSize))*randn(3,3,featureSize,1,'single'), zeros(1,1,'single')}}, ...
    'stride', 1, ...
    'learningRate',[1 0.5], ...
    'dilate',1, ...
    'weightDecay',weightDecay, ...
    'pad', 1, 'opts', {{}}) ;

net.layers{end+1} = struct('type', 'loss') ; % make sure the new 'vl_nnloss.m' is in the same folder.

% Fill in default values
net = vl_simplenn_tidy(net);



function A = clipping(A,b)
A(A>=0&A<b) = b;
A(A<0&A>-b) = -b;




