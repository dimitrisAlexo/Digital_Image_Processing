% function to white balance, demosaic and transform the RAW image

function [Csrgb,Clinear,Cxyz,Ccam] = dng2rgb(rawim,XYZ2Cam,wbcoeffs,bayertype,method,M,N)
    % RAWIM, XYZ2CAM, WBCOEFFS: the outputs of the readdng function
    % BAYERTYPE: bayer pattern {BGGR, GBRG, GRBG, RGGB}
    % METHOD: method of interpolation {nearest, bilinear}
    % M, N: dimensions of new images (MxN)
    % CSRGB: final image after non-linear adjustment for each channel
    % CLINEAR: Cxyz image after transformation according to the CIE protocol
    % CXYZ: Ccam image after transformation according to XYZ2Cam table
    % extracted from the raw image's metadata
    % CCAM: raw image after interpolation
    
    % White balancing
    mask = wbmask(size(rawim,1),size(rawim,2),wbcoeffs,bayertype);
    wbim = rawim.*mask;

    % Dimensions of the original image
    [M0,N0] = size(wbim);

    % Perform Bayer pattern interpolation based on the METHOD variable
    switch lower(method)
        case 'nearest'
            Ccam = nearest(wbim,bayertype,M0,N0,M,N);
        case 'linear'
            Ccam = bilinear(wbim,bayertype,M0,N0,M,N);
        otherwise
            error('Invalid interpolation method. Please choose "nearest" or "linear".');
    end
    
    XYZ2rgb = [3.2406 -1.5372 -0.4986; -0.9689 1.8758 0.0415; 0.0557 -0.2040 1.0570];
    
    cam2linear = XYZ2rgb*XYZ2Cam^-1;  % Assuming previously defined matrices
    cam2linear = cam2linear./repmat(sum(cam2linear,2),1,3);  % Normalize rows to 1
    
    XYZ2Cam = XYZ2Cam./repmat(sum(XYZ2Cam,2),1,3);  % Normalize rows to 1
    Cxyz = transform(Ccam,XYZ2Cam^-1);
    Cxyz = max(0,min(Cxyz,1));  % Always keep image clipped b/w 0-1

    Clinear = transform(Ccam,cam2linear);
    Clinear = max(0,min(Clinear,1));  % Always keep image clipped b/w 0-1
    
    % Fix brightness of Clinear
    grayim = rgb2gray(Clinear);
    grayscale = 0.25/mean(grayim(:));
    Clinear = min(1,Clinear*grayscale);
    Clinear = max(0,min(Clinear,1));  % Always keep image clipped b/w 0-1

    
    % Transform Clinear to Csrgb using non-linear adjustment
    Csrgb = Clinear.^(1/2.2);
    Csrgb = max(0,min(Csrgb,1));  % Always keep image clipped b/w 0-1
    
end

function colormask = wbmask(m,n,wbcoeffs,bayertype)
    % Makes a white-balance multiplicative mask for an image of size m-by-n
    % with RGB while balance multipliers WBCOEFFS = [R_scale G_scale B_scale].
    % BAYERTYPE is string indicating Bayer arrangement: ’rggb’,’gbrg’,’grbg’,’bggr’
    
    colormask = wbcoeffs(2)*ones(m,n); %Initialize to all green values
    switch lower(bayertype)
        case 'rggb'
            colormask(1:2:end,1:2:end) = wbcoeffs(1); %r
            colormask(2:2:end,2:2:end) = wbcoeffs(3); %b
        case 'bggr'
            colormask(2:2:end,2:2:end) = wbcoeffs(1); %r
            colormask(1:2:end,1:2:end) = wbcoeffs(3); %b
        case 'grbg'
            colormask(1:2:end,2:2:end) = wbcoeffs(1); %r
            colormask(2:2:end,1:2:end) = wbcoeffs(3); %b
        case 'gbrg'
            colormask(2:2:end,1:2:end) = wbcoeffs(1); %r
            colormask(1:2:end,2:2:end) = wbcoeffs(3); %b
    end
end

function corrected = transform(im,cmatrix)
    % Applies CMATRIX to RGB input IM. Finds the appropriate weighting of the
    % old color planes to form the new color planes, equivalent to but much
    % more efficient than applying a matrix transformation to each pixel.
    
    if size(im,3)~=3
        error('Apply cmatrix to RGB image only')
    end
    r = cmatrix(1,1)*im(:,:,1)+cmatrix(1,2)*im(:,:,2)+cmatrix(1,3)*im(:,:,3);
    g = cmatrix(2,1)*im(:,:,1)+cmatrix(2,2)*im(:,:,2)+cmatrix(2,3)*im(:,:,3);
    b = cmatrix(3,1)*im(:,:,1)+cmatrix(3,2)*im(:,:,2)+cmatrix(3,3)*im(:,:,3);
    corrected = cat(3,r,g,b);
end

function Ccam = nearest(wbim,bayertype,M0,N0,M,N)

    % Define the step for each dimension
    mstep = M0/M;
    nstep = N0/N;
    
    % Initialize Ccam
    Ccam = zeros(M, N, 3);
    
    for m = 1:1:M
       for n = 1:1:N
           
           % Define the coordinates that correspond to the initial matrix wbim
           m0 = round(1+(m-1)*mstep);
           n0 = round(1+(n-1)*nstep);
           
           oddm0 = mod(m0,2);  % if zero, m0 is even
           oddn0 = mod(n0,2);  % if zero, n0 is even

           switch lower(bayertype)
               case 'bggr'
                   if ~oddm0 && ~oddn0
                       r = wbim(m0,n0);
                       g = wbim(m0,n0-1);
                       b = wbim(m0-1,n0-1);
                   elseif ~oddm0
                       r = wbim(m0,n0+1);
                       g = wbim(m0,n0);
                       b = wbim(m0-1,n0);
                   elseif ~oddn0
                       r = wbim(m0+1,n0);
                       g = wbim(m0,n0);
                       b = wbim(m0,n0-1);
                   else
                       r = wbim(m0+1,n0+1);
                       g = wbim(m0,n0+1);
                       b = wbim(m0,n0);
                   end
               case 'gbrg'
                   if ~oddm0 && ~oddn0
                       r = wbim(m0,n0-1);
                       g = wbim(m0,n0);
                       b = wbim(m0-1,n0);
                   elseif ~oddm0
                       r = wbim(m0,n0);
                       g = wbim(m0,n0+1);
                       b = wbim(m0-1,n0+1);
                   elseif ~oddn0
                       r = wbim(m0+1,n0-1);
                       g = wbim(m0,n0-1);
                       b = wbim(m0,n0);
                   else
                       r = wbim(m0+1,n0);
                       g = wbim(m0,n0);
                       b = wbim(m0,n0+1);
                   end
               case 'grbg'
                   if ~oddm0 && ~oddn0
                       r = wbim(m0-1,n0);
                       g = wbim(m0,n0);
                       b = wbim(m0,n0-1);
                   elseif ~oddm0
                       r = wbim(m0-1,n0+1);
                       g = wbim(m0,n0+1);
                       b = wbim(m0,n0);
                   elseif ~oddn0
                       r = wbim(m0,n0);
                       g = wbim(m0,n0-1);
                       b = wbim(m0+1,n0-1);
                   else
                       r = wbim(m0,n0+1);
                       g = wbim(m0,n0);
                       b = wbim(m0+1,n0);
                   end
               case 'rggb'
                   if ~oddm0 && ~oddn0
                       r = wbim(m0-1,n0-1);
                       g = wbim(m0,n0-1);
                       b = wbim(m0,n0);
                   elseif ~oddm0
                       r = wbim(m0-1,n0);
                       g = wbim(m0,n0);
                       b = wbim(m0,n0+1);
                   elseif ~oddn0
                       r = wbim(m0,n0-1);
                       g = wbim(m0,n0);
                       b = wbim(m0+1,n0);
                   else
                       r = wbim(m0,n0);
                       g = wbim(m0,n0+1);
                       b = wbim(m0+1,n0+1);
                   end
               otherwise
                   error('Invalid Bayer pattern type. Please choose from {BGGR, GBRG, GRBG, RGGB}.');
           end
           
           % Fill the pixel's rgb values
           Ccam(m, n, :) = [r, g, b];
           
       end
    end
end

function Ccam = bilinear(wbim,bayertype,M0,N0,M,N)

    % Define the step for each dimension
    mstep = M0/M;
    nstep = N0/N;
    
    % Initialize Ccam
    Ccam = zeros(M, N, 3);
    
    for m = 1:1:M
       for n = 1:1:N
           
           % Define the coordinates that correspond to the initial matrix wbim
           m0 = round(1+(m-1)*mstep);
           n0 = round(1+(n-1)*nstep);
           
           oddm0 = mod(m0,2);  % if zero, m0 is even
           oddn0 = mod(n0,2);  % if zero, n0 is even

           switch lower(bayertype)
               case 'bggr'
                   if m0 == 1 && n0 == 1
                       r = wbim(m0+1,n0+1);
                       g = 1/2*(wbim(m0+1,n0) + wbim(m0,n0+1));
                       b = wbim(m0,n0);
                   elseif m0 == 1 && oddn0
                       r = 1/2*(wbim(m0+1,n0-1) + wbim(m0+1,n0+1));
                       g = 1/3*(wbim(m0,n0-1) + wbim(m0,n0+1) + wbim(m0+1,n0));
                       b = wbim(m0,n0);
                   elseif m0 == 1 && ~oddn0
                       r = wbim(m0+1,n0);
                       g = wbim(m0,n0);
                       b = 1/2*(wbim(m0,n0-1) + wbim(m0,n0+1));
                   elseif n0 == 1 && oddm0
                       r = 1/2*(wbim(m0-1,n0+1) + wbim(m0+1,n0+1));
                       g = 1/3*(wbim(m0-1,n0) + wbim(m0,n0+1) + wbim(m0+1,n0));
                       b = wbim(m0,n0);
                   elseif n0 == 1 && ~oddm0
                       r = wbim(m0,n0+1);
                       g = wbim(m0,n0);
                       b = 1/2*(wbim(m0-1,n0) + wbim(m0+1,n0));
                   elseif ~oddm0 && ~oddn0
                       r = wbim(m0,n0);
                       g = 1/4*(wbim(m0-1,n0) + wbim(m0,n0-1) + wbim(m0,n0+1) + wbim(m0+1,n0));
                       b = 1/4*(wbim(m0-1,n0-1) + wbim(m0-1,n0+1) + wbim(m0+1,n0-1) + wbim(m0+1,n0+1));
                   elseif ~oddm0
                       r = 1/2*(wbim(m0,n0-1) + wbim(m0,n0+1));
                       g = wbim(m0,n0);
                       b = 1/2*(wbim(m0-1,n0) + wbim(m0+1,n0));
                   elseif ~oddn0
                       r = 1/2*(wbim(m0-1,n0) + wbim(m0+1,n0));
                       g = wbim(m0,n0);
                       b = 1/2*(wbim(m0,n0-1) + wbim(m0,n0+1));
                   else
                       r = 1/4*(wbim(m0-1,n0-1) + wbim(m0-1,n0+1) + wbim(m0+1,n0-1) + wbim(m0+1,n0+1));
                       g = 1/4*(wbim(m0-1,n0) + wbim(m0,n0-1) + wbim(m0,n0+1) + wbim(m0+1,n0));
                       b = wbim(m0,n0);
                   end
               case 'gbrg'
                   if m0 == 1 && n0 == 1
                       r = wbim(m0+1,n0);
                       g = wbim(m0,n0);
                       b = wbim(m0,n0+1);
                   elseif m0 == 1 && oddn0
                       r = wbim(m0+1,n0);
                       g = wbim(m0,n0);
                       b = 1/2*(wbim(m0,n0-1) + wbim(m0,n0+1));
                   elseif m0 == 1 && ~oddn0
                       r = 1/2*(wbim(m0+1,n0-1) + wbim(m0+1,n0+1));
                       g = 1/3*(wbim(m0,n0-1) + wbim(m0,n0+1) + wbim(m0+1,n0));
                       b = wbim(m0,n0);
                   elseif n0 == 1 && oddm0
                       r = 1/2*(wbim(m0-1,n0) + wbim(m0+1,n0));
                       g = wbim(m0,n0);
                       b = wbim(m0,n0+1);
                   elseif n0 == 1 && ~oddm0
                       r = wbim(m0,n0);
                       g = 1/3*(wbim(m0-1,n0) + wbim(m0,n0+1) + wbim(m0+1,n0));
                       b = 1/2*(wbim(m0-1,n0+1) + wbim(m0+1,n0+1));
                   elseif ~oddm0 && ~oddn0
                       r = 1/2*(wbim(m0,n0-1) + wbim(m0,n0+1));
                       g = wbim(m0,n0);
                       b = 1/2*(wbim(m0-1,n0) + wbim(m0+1,n0));
                   elseif ~oddm0
                       r = wbim(m0,n0);
                       g = 1/4*(wbim(m0-1,n0) + wbim(m0,n0-1) + wbim(m0,n0+1) + wbim(m0+1,n0));
                       b = 1/4*(wbim(m0-1,n0-1) + wbim(m0-1,n0+1) + wbim(m0+1,n0-1) + wbim(m0+1,n0+1));
                   elseif ~oddn0
                       r = 1/4*(wbim(m0-1,n0-1) + wbim(m0-1,n0+1) + wbim(m0+1,n0-1) + wbim(m0+1,n0+1));
                       g = 1/4*(wbim(m0-1,n0) + wbim(m0,n0-1) + wbim(m0,n0+1) + wbim(m0+1,n0));
                       b = wbim(m0,n0);
                   else
                       r = 1/2*(wbim(m0-1,n0) + wbim(m0+1,n0));
                       g = wbim(m0,n0);
                       b = 1/2*(wbim(m0,n0-1) + wbim(m0,n0+1));
                   end
               case 'grbg'
                   if m0 == 1 && n0 == 1
                       r = wbim(m0,n0+1);
                       g = wbim(m0,n0);
                       b = wbim(m0+1,n0);
                   elseif m0 == 1 && oddn0
                       r = 1/2*(wbim(m0,n0-1) + wbim(m0,n0+1));
                       g = wbim(m0,n0);
                       b = wbim(m0+1,n0);
                   elseif m0 == 1 && ~oddn0
                       r = wbim(m0,n0);
                       g = 1/3*(wbim(m0,n0-1) + wbim(m0,n0+1) + wbim(m0+1,n0));
                       b = 1/2*(wbim(m0+1,n0-1) + wbim(m0+1,n0+1));
                   elseif n0 == 1 && oddm0
                       r = wbim(m0,n0+1);
                       g = wbim(m0,n0);
                       b = 1/2*(wbim(m0-1,n0) + wbim(m0+1,n0));
                   elseif n0 == 1 && ~oddm0
                       r = 1/2*(wbim(m0-1,n0+1) + wbim(m0+1,n0+1));
                       g = 1/3*(wbim(m0-1,n0) + wbim(m0,n0+1) + wbim(m0+1,n0));
                       b = wbim(m0,n0);
                   elseif ~oddm0 && ~oddn0
                       r = 1/2*(wbim(m0-1,n0) + wbim(m0+1,n0));
                       g = wbim(m0,n0);
                       b = 1/2*(wbim(m0,n0-1) + wbim(m0,n0+1));
                   elseif ~oddm0
                       r = 1/4*(wbim(m0-1,n0-1) + wbim(m0-1,n0+1) + wbim(m0+1,n0-1) + wbim(m0+1,n0+1));
                       g = 1/4*(wbim(m0-1,n0) + wbim(m0,n0-1) + wbim(m0,n0+1) + wbim(m0+1,n0));
                       b = wbim(m0,n0);
                   elseif ~oddn0
                       r = wbim(m0,n0);
                       g = 1/4*(wbim(m0-1,n0) + wbim(m0,n0-1) + wbim(m0,n0+1) + wbim(m0+1,n0));
                       b = 1/4*(wbim(m0-1,n0-1) + wbim(m0-1,n0+1) + wbim(m0+1,n0-1) + wbim(m0+1,n0+1));
                   else
                       r = 1/2*(wbim(m0,n0-1) + wbim(m0,n0+1));
                       g = wbim(m0,n0);
                       b = 1/2*(wbim(m0-1,n0) + wbim(m0+1,n0));
                   end
               case 'rggb'
                   if m0 == 1 && n0 == 1
                       r = wbim(m0,n0);
                       g = 1/2*(wbim(m0+1,n0) + wbim(m0,n0+1));
                       b = wbim(m0+1,n0+1);
                   elseif m0 == 1 && oddn0
                       r = wbim(m0,n0);
                       g = 1/3*(wbim(m0,n0-1) + wbim(m0,n0+1) + wbim(m0+1,n0));
                       b = 1/2*(wbim(m0+1,n0-1) + wbim(m0+1,n0+1));
                   elseif m0 == 1 && ~oddn0
                       r = 1/2*(wbim(m0,n0-1) + wbim(m0,n0+1));
                       g = wbim(m0,n0);
                       b = wbim(m0+1,n0);
                   elseif n0 == 1 && oddm0
                       r = wbim(m0,n0);
                       g = 1/3*(wbim(m0-1,n0) + wbim(m0,n0+1) + wbim(m0+1,n0));
                       b = 1/2*(wbim(m0-1,n0+1) + wbim(m0+1,n0+1));
                   elseif n0 == 1 && ~oddm0
                       r = 1/2*(wbim(m0-1,n0) + wbim(m0+1,n0));
                       g = wbim(m0,n0);
                       b = wbim(m0,n0+1);
                   elseif ~oddm0 && ~oddn0
                       r = 1/4*(wbim(m0-1,n0-1) + wbim(m0-1,n0+1) + wbim(m0+1,n0-1) + wbim(m0+1,n0+1));
                       g = 1/4*(wbim(m0-1,n0) + wbim(m0,n0-1) + wbim(m0,n0+1) + wbim(m0+1,n0));
                       b = wbim(m0,n0);
                   elseif ~oddm0
                       r = 1/2*(wbim(m0-1,n0) + wbim(m0+1,n0));
                       g = wbim(m0,n0);
                       b = 1/2*(wbim(m0,n0-1) + wbim(m0,n0+1));
                   elseif ~oddn0
                       r = 1/2*(wbim(m0,n0-1) + wbim(m0,n0+1));
                       g = wbim(m0,n0);
                       b = 1/2*(wbim(m0-1,n0) + wbim(m0+1,n0));
                   else
                       r = wbim(m0,n0);
                       g = 1/4*(wbim(m0-1,n0) + wbim(m0,n0-1) + wbim(m0,n0+1) + wbim(m0+1,n0));
                       b = 1/4*(wbim(m0-1,n0-1) + wbim(m0-1,n0+1) + wbim(m0+1,n0-1) + wbim(m0+1,n0+1));
                   end
               otherwise
                   error('Invalid Bayer pattern type. Please choose from {BGGR, GBRG, GRBG, RGGB}.');
           end
           
           % Fill the pixel's rgb values
           Ccam(m, n, :) = [r, g, b];
           
       end
    end
end

