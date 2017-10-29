# CSNet
This is reimplemenation of CSNet [1] for block based compressive sensing reconstruction. CSNet is implemented in Matconvnet. 
This implement is motivated by DnCNN implementation [2]


## Current Performance | PSNR (dB)
                              GSR		      Org Reported 	       ReImplement     Best(17 conv, adapt learn rate) 
| Image 	|Rate	|PSRN	|SSIM	|PSNR	|SSIM	|PSNR    |SSIM    |PSNR	|SSIM |
| ---     | ---     |---      |---      | ---     |---      | ---    |---     | ---  |---|
| baby	|0.1		|32.18	|0.8832	|34.83	|0.9170	|33.36   |0.902   |33.75	|0.907|
| bird 	|0.1		|34.47	|0.9411	|35.15	|0.9476	|33.05   |0.931   |34.47	|0.949|
| butter	|0.1		|23.78	|0.8279	|28.01	|0.9018	|25.71   |0.859   |27.53	|0.914|
| Avg	|		|30.14    |0.8841	|32.66	|0.9221	|30.71   |0.897   |31.91	|0.923|


## How to run
In order to train the CSNet from the scratch, you should run 
1. 'GenerateTrainingPatches.m' first. It will create trainding data outsize of this CSNet folder (for 100Mb limitation of github). 

2. TrainingCode/CSNet_v02/Demo_Train.m Training data is saved in "data/CSNet<noLayer>_r<subrate>_blk<block_size>_mBat<no_mini_batch_size>_<isLearnSamplingMatrix>_<isLearnBiasSampling>"
  

## Disclaimer 
Due to some parameters are not mentioned in [1], I try my best to reproduce the resported results, by evaluating several parameter. However, the re-implementation results (PSNR - dB) are still 1~2dB lower than reported results. 

If you find the better configurations, or any suggestion. Feeling free to recommend me. 


## Reference
[1] S. Wuzhen et al, â€œDeep network for compressed image sensing.â€? IEEE Inter. Conf. Multimedia Expo, Jul-2017.

[2] K. Zhang et al, Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising, available at https://github.com/cszn/DnCNN

