clearvars
clc
warning off

testimage = "RawImage.DNG";
bayertype = 'RGGB';  % choose from {BGGR, GBRG, GRBG, RGGB}
method = 'linear';  % choose from {nearest, linear}
M = 1348;
N = 2196;

[rawim,XYZ2Cam,wbcoeffs] = readdng(testimage);
[Csrgb,Clinear,Cxyz,Ccam] = dng2rgb(rawim,XYZ2Cam,wbcoeffs,bayertype,method,M,N);

rgb = 'RGB';

% Show rawim and its histograms
figure(1)
imshow(rawim)
title('Raw Image')

% Show Ccam and its histograms
figure(2)
subplot(2,2,1)
imshow(Ccam)
title('Ccam Image')
for i=1:3
    subplot(2,2,i+1)
    histogram(Ccam(:,:,i))
    xlabel('Pixel Value')
    ylabel('Frequency')
    title(sprintf('%s channel', rgb(i)))
end

% Show Cxyz and its histograms
figure(3)
subplot(2,2,1)
imshow(Cxyz)
title('Cxyz Image')
for i=1:3
    subplot(2,2,i+1)
    histogram(Cxyz(:,:,i))
    xlabel('Pixel Value')
    ylabel('Frequency')
    title(sprintf('%s channel', rgb(i)))
end

% Show Clinear and its histograms
figure(4)
subplot(2,2,1)
imshow(Clinear)
title('Clinear Image')
for i=1:3
    subplot(2,2,i+1)
    histogram(Clinear(:,:,i))
    xlabel('Pixel Value')
    ylabel('Frequency')
    title(sprintf('%s channel', rgb(i)))
end

% Show Csrgb and its histograms
figure(5)
subplot(2,2,1)
imshow(Csrgb)
title('Csrgb Image')
for i=1:3
    subplot(2,2,i+1)
    histogram(Csrgb(:,:,i))
    xlabel('Pixel Value')
    ylabel('Frequency')
    title(sprintf('%s channel', rgb(i)))
end
