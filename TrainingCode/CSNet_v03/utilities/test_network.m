
% function net = CSNet_init
global featureSize noLayer blkSize subRate;

test = 1;
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

% net.layers{end+1} = struct('type', 'conv', ...
%     'weights', {{zeros(blkSize, blkSize, 1, noMeas,'single'), zeros(featureSize,1,'single')}}, ...
%     'stride', blkSize, ...
%     'pad', 0, ...
%     'dilate',1, ...
%     'learningRate',lr00, ...
%     'weightDecay',weightDecay, ...
%     'opts',{{}}) ;
% % net.layers{end+1} = struct('type', 'relu','leak',0) ; -- do not use relu 

% assign the sampling matrix
W = zeros(blkSize, blkSize, 1, noMeas); 
for i = 1:1:noMeas
    W(:, :, 1, i) = reshape(Phi(i, :), blkSize, blkSize); 
end

% x  = zeros(256, 256, 1, 2); 
im = double(imread('cameraman.tif'));
im = im(1:256, 1:192);
% x(:, :, 1, 1) = im; 
% x(:, :, 1, 2) = im ; 

% First convolution layer
x1 = vl_nnconv(im, W, [], 'stride', blkSize);

% second convolution layer
%W2 = sqrt(2/ (noMeas * blkSize^2)) * randn(1, 1, noMeas, blkSize* blkSize); 
W2 = zeros(1, 1, noMeas, blkSize*blkSize); 
PhiInv = pinv(Phi); 
for i = 1:1:noMeas
    W2(:, :, i, :) = PhiInv(:, i); 
end
x2 = vl_nnconv(x1, W2, []); 

for i = 1:1:size(im, 1)/blkSize
    for j = 1:1:size(im, 2)/blkSize
        tmpPatch = im( (i-1)*blkSize + 1:i*blkSize, (j-1)*blkSize +1 :j*blkSize); 
        x2(i, j, :) = tmpPatch(:); 
    end
end

x3 = vl_nnreshapeconcat(x2, []); imshow(x3, []); 

%% Verify backward propagation 
p = x3; %rand(size(x3), 'single'); 
[dx2] = vl_nnreshapeconcat(x2, p);

x3 = vl_nnreshapeconcat(dx2, []); figure(2); imshow(x3, []); 


%% 2. Initial reconstruction layer with 1x1 Convolution 
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{sqrt(2/(1*blkSize*blkSize))*randn(1, 1, noMeas, blkSize*blkSize,'single'), zeros(featureSize,1,'single')}}, ...
    'stride', 1, ...
    'pad', 0, ...
    'dilate',1, ...
    'learningRate',lr11, ...
    'weightDecay',weightDecay, ...
    'opts',{{}}) ;

%% 3. Reshape and concatinate to make recon. image 
%net = addCustomLayer(net, @reshapeForward, @reshapeBackward); 
net.layers{end+1} = struct{'type', 'reshape'};

% Concatinate t
net.layers{end+1} = struct{'type', 'concat'};

%% 4. Reconstruction network - DnCNN 
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{sqrt(2/(9*featureSize))*randn(3,3,1,featureSize,'single'), zeros(featureSize,1,'single')}}, ...
    'stride', 1, ...
    'pad', 1, ...
    'dilate',1, ...
    'learningRate',lr11, ...
    'weightDecay',weightDecay, ...
    'opts',{{}}) ;
net.layers{end+1} = struct('type', 'relu','leak',0) ;

for i = 1:1:noLayer - 2
    
    net.layers{end+1} = struct('type', 'conv', ...
        'weights', {{sqrt(2/(9*featureSize))*randn(3,3,featureSize,featureSize,'single'), zeros(featureSize,1,'single')}}, ...
        'stride', 1, ...
        'learningRate',lr10, ...
        'dilate',1, ...
        'weightDecay',weightDecay, ...
        'pad', 1, 'opts', {{}}) ;
    
    net.layers{end+1} = struct('type', 'bnorm', ...
        'weights', {{clipping(sqrt(2/(9*featureSize))*randn(featureSize,1,'single'),b_min), zeros(featureSize,1,'single'),meanvar}}, ...
        'learningRate', [1 1 1], ...
        'weightDecay', [0 0], ...
        'opts', {{}}) ;
    net.layers{end+1} = struct('type', 'relu','leak',0) ;
    
end

net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{sqrt(2/(9*featureSize))*randn(3,3,featureSize,1,'single'), zeros(1,1,'single')}}, ...
    'stride', 1, ...
    'learningRate',lr11, ...
    'dilate',1, ...
    'weightDecay',weightDecay, ...
    'pad', 1, 'opts', {{}}) ;

net.layers{end+1} = struct('type', 'loss') ; % make sure the new 'vl_nnloss.m' is in the same folder.

% Fill in default values
net = vl_simplenn_tidy(net);



% function A = clipping(A,b)
% A(A>=0&A<b) = b;
% A(A<0&A>-b) = -b;




