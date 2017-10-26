# CSNet
This is reimplemenation of CSNet [1] for block based compressive sensing reconstruction. CSNet is implemented in Matconvnet. 
This implement is motivated by DnCNN implementation [2]


## Current Performance | PSNR (dB)


## How to run
In order to train the CSNet from the scratch, you should run 
1. 'GenerateTrainingPatches.m' first. It will create trainding data outsize of this CSNet folder (for 100Mb limitation of github). 

2. TrainingCode/CSNet_v02/Demo_Train.m Training data is saved in "data/CSNet<noLayer>_r <subrate>_blk<block_size>_mBat<no_mini_batch_size>_<isLearnSamplingMatrix>_<isLearnBiasSampling>"
  

## Disclaimer 
Due to some parameters are not mentioned in [1], I try my bet to reproduce the resported results, by evaluating several parameter. However, the re-implementation results (PSNR - dB) are still 1~2dB lower than reported results. 

If you find the better configurations, or any suggestion. Feeling free to recommend me. 


## Reference
[1] S. Wuzhen et al, “Deep network for compressed image sensing.” IEEE Inter. Conf. Multimedia Expo, Jul-2017.

[2] K. Zhang et al, Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising, available at https://github.com/cszn/DnCNN

