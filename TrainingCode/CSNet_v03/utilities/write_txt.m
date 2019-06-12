function write_txt(fileName, imgName, subRate, PSNRCur, SSIMCur )

fileID = fopen(fileName,'w');
fprintf(fileID,'Img   subrate    PSNR         SSIM\n');
for i = 1:1:length(PSNRCur)
    fprintf(fileID,'%s \t %6.3f \t %6.3f \t %6.3f \n', imgName{i}, subRate, PSNRCur(i), SSIMCur(i));
end

fprintf(fileID,'Avg  %6.3f \t %6.3f \t %6.3f \n', subRate, mean(PSNRCur), mean(SSIMCur));

fclose(fileID);