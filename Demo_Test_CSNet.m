addpath('E:\matConvNet\matconvnet-1.0-beta25\matlab\mex');
addpath('E:\matConvNet\matconvnet-1.0-beta25\matlab\simplenn');
% addpath('E:\matConvNet\matconvnet-1.0-beta25\matlab');
addpath('utilities'); 

%%% test the model performance

% clear; clc;
format compact;
global featureSize noLayer subRate blkSize isLearnMtx; %%% noise level

featureSize = 64;
noLayer = 17;
subRate = 0.1;
blkSize = 32;
isLearnMtx = [1 0];


batSize = 64;
addpath(fullfile('Data','utilities'));
folderTest  = 'Set5'; %%% test dataset
%folderTest  = fullfile('data','Test','Set68'); %%% test dataset

showResult  = 0;
writeRecon  = 1; 
useGPU      = 1;
pauseTime   = 5;

epoch       = 100;

modelName   = ['Orgv3_CSNet' num2str(noLayer) '_' num2str(featureSize) '_r' num2str(subRate) ...
    '_blk' num2str(blkSize) '_mBat' num2str(batSize) ...
    '_' num2str(isLearnMtx(1)) '_' num2str(isLearnMtx(2)) ...
    '-epoch-',num2str(epoch)]; %%% model name

%%% load Gaussian denoising model
load(fullfile('models', [modelName, '.mat']));

net = vl_simplenn_tidy(net);
net.layers = net.layers(1:end-1);

%%%
net = vl_simplenn_tidy(net);

% for i = 1:size(net.layers,2)
%     net.layers{i}.precious = 1;
% end

%%% move to gpu
if useGPU
    net = vl_simplenn_move(net, 'gpu') ;
end

%%% read images
ext         =  {'*.jpg','*.png','*.bmp'};
filePaths   =  [];
for i = 1 : length(ext)
    filePaths = cat(1,filePaths, dir(fullfile('testsets',folderTest,ext{i})));
end

%%% PSNR and SSIM
PSNRs = zeros(1,1);
SSIMs = zeros(1,1);

count = 1;
allName = cell(1);
for i = 1:length(filePaths)
    
    %%% read images
    label = imread(fullfile('testsets', folderTest, filePaths(i).name));
    [~,nameCur,extCur] = fileparts(filePaths(i).name);
    
    if size(label,3)==3
        label = rgb2gray(label);
    end
    
    if mod(size(label, 1), blkSize) ~= 0 || mod(size(label, 2), blkSize) ~= 0
        continue
    end
    
    allName{count} = nameCur;
    
    label = im2double(label);
    
    input = single(label);
    
    %%% convert to GPU
    if useGPU
        input = gpuArray(input);
    end
    
    res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test');
    %output = input - res(end).x;
    output = res(end).x;
    %%% convert to CPU
    if useGPU
        output = gather(output);
        input  = gather(input);
    end
    
    %%% calculate PSNR and SSIM
    [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(label),im2uint8(output),0,0);
    if showResult
        imshow(cat(2,im2uint8(label),im2uint8(input),im2uint8(output)));
        title([filePaths(i).name,'    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
        drawnow;
        pause(pauseTime)
    end
    PSNRs(count) = PSNRCur;
    SSIMs(count) = SSIMCur;
    
    
    % save results for current image
    if writeRecon
        folder  = ['Results\2Image_CSNet' num2str(noLayer) '_' num2str(featureSize) ...
            '_blk' num2str(blkSize) '_mBat' num2str(batSize) ...
             '_' num2str(isLearnMtx(1)) '_' num2str(isLearnMtx(2)) '_epoch' num2str(epoch)];
        if ~exist(folder), mkdir(folder); end
        fileName = [folder '\' folderTest '_' allName{count} '_subrate' num2str(subRate) '.png'];
        imwrite(im2uint8(output), fileName );
        count = count + 1;
    end
end

% save results for current image
folder  = ['Results\1Text_CSNet' num2str(noLayer) '_' num2str(featureSize) ...
    '_blk' num2str(blkSize) '_mBat' num2str(batSize) ...
    '_' num2str(isLearnMtx(1)) '_' num2str(isLearnMtx(2)) '_epoch' num2str(epoch)];
if ~exist(folder), mkdir(folder); end
imgName = [folderTest ];
fileName = [folder '\' imgName '_subrate' num2str(subRate) '.txt'];
write_txt(fileName, allName, subRate, PSNRs, SSIMs );

disp([mean(PSNRs),mean(SSIMs)]);




