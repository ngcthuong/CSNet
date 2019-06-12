function image = data_augmentation_CSNet(image, mode)
% image = imread('cameraman.tif');
% mode = 5; 
if mode == 1
    return;
end

if mode == 2 % flipped
    image = flipud(image);
    return;
end

if mode == 3 % rotation 90
    image = fliplr(image);
    return;
end

if mode == 4 % rotation 90 & flipped
    image = fliplr(image);
    image = flipud(image);
    return;
end
% imshow(image, []); 
% title(['Mode ' num2str(mode)]); 
