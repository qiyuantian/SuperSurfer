%% Qiyuan Tian 2022
clear, clc, close all

dpRoot = fileparts(which('s_t1wSim.m'));
dpData = fullfile(dpRoot, 'data');
dpSim = fullfile(dpRoot, 'data-sim');
mkdir(dpSim);

%% file pre
files = dir(fullfile(dpData, 'hcp*'))';

%% downsample resolution
for ii = 1 : length(files)
    
    fnFile = files(ii).name;
    fpFile = fullfile(dpData, fnFile);
    
    tmp = load(fpFile);
    t1w = tmp.t1w;
    mask = tmp.mask; % find area of interest
    maskdil = tmp.mask_dilate;
    
    %%% simulate low resolution data, 0.7 of original resolution
    sz = size(t1w); % here assume sizes are odd
    sz_new = round(sz * 0.5 / 2) * 2 + 1; % hann zone
    sz_pad = (sz - sz_new) / 2; % zero zone
    
    ker1 = hann(sz_new(1));
    sz_repmat1 = sz_new; sz_repmat1(1) = 1; ker1_3d = repmat(ker1, sz_repmat1);
    
    ker2 = hann(sz_new(2)); ker2 = reshape(ker2, [1, length(ker2)]);
    sz_repmat2 = sz_new; sz_repmat2(2) = 1; ker2_3d = repmat(ker2, sz_repmat2);
    
    ker3 = hann(sz_new(3)); ker3 = reshape(ker3, [1, 1, length(ker3)]);
    sz_repmat3 = sz_new; sz_repmat3(3) = 1; ker3_3d = repmat(ker3, sz_repmat3);
    
    ker_3d = zeros(size(t1w));
    ker_3d(sz_pad(1)+1:sz_pad(1)+sz_new(1), sz_pad(2)+1:sz_pad(2)+sz_new(2), sz_pad(3)+1:sz_pad(3)+sz_new(3)) ...
              = ker1_3d .* ker2_3d .* ker3_3d; %in each dimmention middle 1/2 is hann, external is 0
    
    kimg = fftshift( fftn( t1w ) );
    kimg_new = kimg .* ker_3d;    
    t1w_lowres = ifftn( ifftshift( kimg_new ) );
    t1w_lowres = t1w_lowres .* maskdil;

    figure, imshow(ker_3d(:, :, 130), [0, 1]);
    figure, imshow(t1w_lowres(:, :, 100), [0, 1200]);
    figure, imshow(t1w(:, :, 100) - t1w_lowres(:, :, 100), [-500, 500]);
    
    %%% save data
    fnSave = [fnFile(1 : end - 4) '_sim.mat'];
    fpSave = fullfile(dpSim, fnSave);
    save(fpSave, 't1w_lowres');
end   


