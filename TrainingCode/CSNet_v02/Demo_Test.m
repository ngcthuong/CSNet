addpath('E:\matConvNet\matconvnet-1.0-beta25\matlab\mex');
addpath('E:\matConvNet\matconvnet-1.0-beta25\matlab\simplenn');
% addpath('E:\matConvNet\matconvnet-1.0-beta25\matlab');

% clear; clc;
format compact;
global featureSize noLayer subRate blkSize isLearnMtx; %%% noise level

featureSize = 64;
noLayer     = 5;
subRate     = 0.1;
blkSize     = 32;
isLearnMtx  = [1 0];
batSize     = 64;

addpath(fullfile('../../Data','utilities'));
dataSet = 'Set5'; 
folderTest  = fullfile('../../','testsets',dataSet); %%% test dataset
%folderTest  = fullfile('data','Test','Set68');1 %%% test dataset

showResult  = 0;
useGPU      = 1;
pauseTime   = 0;

epoch       = 83;
modelName   = ['CSNet' num2str(noLayer) '_' num2str(featureSize) '_r' num2str(subRate) ...
    '_blk' num2str(blkSize) '_mBat' num2str(batSize) ...
    '_' num2str(isLearnMtx(1)) '_' num2str(isLearnMtx(2)) ]; %%% model name


for iter = 1:1:epoch
    
    %%% load Gaussian denoising model
    load(fullfile('data',modelName,[modelName,'-epoch-',num2str(iter),'.mat']));
    net = vl_simplenn_tidy(net);
    net.layers = net.layers(1:end-1);
    
    %%%
    net = vl_simplenn_tidy(net);    
       
    %%% move to gpu
    if useGPU
        net = vl_simplenn_move(net, 'gpu') ;
    end
    
    %%% read images
    ext         =  {'*.jpg','*.png','*.bmp'};
    filePaths   =  [];
    for i = 1 : length(ext)
        filePaths = cat(1,filePaths, dir(fullfile(folderTest,ext{i})));
    end
    
    %%% PSNR and SSIM
    PSNRs = zeros(1,1);
    SSIMs = zeros(1,1);
    
    for i = 1:length(filePaths)
        
        %%% read images
        label = imread(fullfile(folderTest,filePaths(i).name));
        [~,nameCur,extCur] = fileparts(filePaths(i).name);
        label = im2double(label);
        
        if size(label,3)==3
            label = rgb2gray(label);
        end
        
        if mod(size(label, 1), blkSize) ~= 0 || mod(size(label, 2), blkSize) ~= 0
            continue
        end
        
        randn('seed',0);
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
        PSNRs(i) = PSNRCur;
        SSIMs(i) = SSIMCur;
    end
    
    disp(['Epoch ' num2str(iter) ', ' num2str(mean(PSNRs)) 'dB, ' num2str(mean(SSIMs))]);
    
    allPSNR(iter) = mean(PSNRs);
    allSSIM(iter) = mean(SSIMs);
    
    
end
pr = figure(1);
plot(allPSNR, 'r', 'LineWidth', 3); xlabel('epoch'); ylabel('PSNR (dB)'); 
title(['Average PSNR for ' dataSet]); grid on;
saveas(pr, ['Results\PSNR_' modelName '.png']);
saveas(pr, ['Results\PSNR_' modelName '.fig']);
pr = figure(2);
plot(allSSIM, 'r', 'LineWidth', 3); xlabel('epoch'); ylabel('SSIM'); 
title(['Average SSIM for ' dataSet]); grid on; 
saveas(pr, ['Results\SSIM_' modelName '.png']);
saveas(pr, ['Results\SSIM_' modelName '.fig']);
