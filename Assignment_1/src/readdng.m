% function to get the useful part of a RAW image and its metadata

function [rawim,XYZ2Cam,wbcoeffs] = readdng(filename)
    % filename: path of a .DNG RAW image
    % rawim: measurements extracted from the sensor (without metadata)
    % XYZ2Cam: 3x3 matrix for camera's colorspace convertion
    % wbcoeffs: 1x3 vector with white balance correction coefficients

    % read RAW image
    warning off MATLAB:tifflib:TIFFReadDirectory:libraryWarning
    obj = Tiff(filename,'r');
    offsets = getTag(obj,'SubIFD');
    setSubDirectory(obj,offsets(1));
    rawim = read(obj);
    close(obj);
    
    % read useful metadata
    meta_info = imfinfo(filename);
    
    % (x_origin,y_origin) is the upper left corner of the useful part of the sensor and consequently of the array rawim
    y_origin = meta_info.SubIFDs{1}.ActiveArea(1)+1;
    x_origin = meta_info.SubIFDs{1}.ActiveArea(2)+1;
    
    % width and height of the image (the useful part of array rawim)
    width = meta_info.SubIFDs{1}.DefaultCropSize(1);
    height = meta_info.SubIFDs{1}.DefaultCropSize(2);
    
    blacklevel = meta_info.SubIFDs{1}.BlackLevel(1);  % sensor value corresponding to black
    whitelevel = meta_info.SubIFDs{1}.WhiteLevel;  % sensor value corresponding to white
    
    wbcoeffs = (meta_info.AsShotNeutral).^-1;
    wbcoeffs = wbcoeffs/wbcoeffs(2);  % green channel will be left unchanged
    XYZ2Cam = meta_info.ColorMatrix2;
    XYZ2Cam = reshape(XYZ2Cam,3,3)';
    
    % extract the useful data from rawim
    rawim = rawim(y_origin:y_origin+height-1, x_origin:x_origin+width-1, :);

    % check for non-linearity
    if isfield(meta_info.SubIFDs{1},'LinearizationTable')
        warning('Non-linear transformation to the sensor data')
        lintable = meta_info.SubIFDs{1}.LinearizationTable;
        rawim = lintable(rawim+1);
    end
    
    % normalize the pixel values to the range [0, 1]
    rawim = double(rawim - blacklevel)/double(whitelevel - blacklevel);

    % clip pixel values below 0 to 0 and above 1 to 1
    rawim = max(0,min(rawim,1));
    
end

