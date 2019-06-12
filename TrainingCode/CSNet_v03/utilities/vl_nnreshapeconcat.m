function y = vl_nnreshapeconcat(x, dzdy)
global blkSize
% VL_NNRESHAPE Feature reshaping
%   Y = VL_NNRESHAPE(X, DIMS) reshapes the input data X to have
%   the dimensions specified by DIMS. X is a SINGLE array of
%   dimension H x W x D x N where (H,W) are the height and width of
%   the map stack, D is the image depth (number of feature channels)
%   and N the number of of images in the stack. DIMS is a 1 x 3 array
%   of integers describing the dimensions that Y will take (batch
%   size is preserved). In addition to positive integers, the
%   following can also be specified in the style of caffe:
%
%   Interpretation of DIMS elements:
%   -1 := work it out from other dims
%    0 := copy dimension from X
%
%   NOTE: At most one dimension can be worked out from the others.
%
%   DZDX = VL_NNRESHAPE(X, DIMS, DZDY) computes the derivatives of the
%   block projected onto DZDY. DZDX and DZDY have the same dimensions
%   as X and Y respectively.


if nargin <= 1 || isempty(dzdy)  % Forward pass
    [w, h, c, b] = size(x);         % input size
    blkSize = sqrt(c);
    dims = [blkSize * blkSize, w * h]; % output size    
    
    y = single(zeros(w* blkSize, h * blkSize, 1, b));
    x = gather(x); 
    for k = 1:1:b
        % Step 1. Reshape
        tmpMtx  = zeros(dims);  
        count = 0;
        for i = 1:1:h
            for j = 1:1:w
                try 
                    count = count + 1; 
                    tmpMtx(:, count) = x(j, i, :,k);
                    
                catch
                    disp('tmMtx Size'); disp(size(tmpMtx)); 
                    disp(['count = ' num2str(count), 'i = ' num2str(i) ...
                        ', j = ' num2str(j) ', k = ' num2str(k)]); 
                end
            end
        end        
        % step 2. Concatiniate
        y(:, :, 1, k) = col2im(double(tmpMtx), [32, 32], ...
            [blkSize * w, blkSize * h], 'distinct');
    end
    y = gpuArray(y); 
else
    
    [w, h, c, b] = size(dzdy); % input sizes 256x256x1xb
    % default block size is 32x32
    y = gpuArray(zeros(w/blkSize, h/blkSize, blkSize * blkSize, b));
    for k = 1:1:b
        blkImg = (im2col(dzdy(:, :, :, k), [blkSize, blkSize], 'distinct'));
        count = 1; 
        for i = 1:1:h/blkSize
            for j = 1:1:w/blkSize
                y(j, i, :, b) = blkImg(:, count); 
                count = count + 1; 
            end
        end
    end
    y = single(y);
end