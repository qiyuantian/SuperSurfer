%% quantify the similarity between ground-truth image volumes and lowres image volumes
clear, clc, close all

% set path
dpRoot = fileparts(which('compare.m'));
dpData = fullfile(dpRoot, 'data', 'hcp001_data.mat');
dpSim = fullfile(dpRoot, 'data-sim', 'hcp001_data_sim.mat');
dpPred = fullfile(dpRoot, 'pred-supersurfer', 'hcp001_supersurfer.mat');
dpSave = fullfile(dpRoot, 'compare');
mkdir(dpSave);

%% file pre
tmp = load(dpData);
t1w = tmp.t1w; % select one picture
mask = tmp.mask;

tmp = load(dpSim);
t1w_lowres = tmp.t1w_lowres;

tmp = load(dpPred);
pred = tmp.img_pred_denorm;

[t1w_lowres_norm, t1w_norm] = norm(t1w_lowres, t1w, mask);
[t1w_lowres_norm, pred_norm] = norm(t1w_lowres, pred, mask);

t1w_lowsnr_diff = t1w_lowres_norm - t1w_norm;
pred_diff = pred_norm - t1w_norm;

%% MAD

t1w_lowsnr_mad = mean(abs(t1w_lowres_norm - t1w_norm),'all'); 
pred_mad = mean(abs(pred_norm - t1w_norm),'all'); 

%% PSNR

t1w_lowsnr_psnr = psnr(t1w_lowres_norm, t1w_norm); 
pred_psnr = psnr(pred_norm, t1w_norm);

%% SSIM

[t1w_lowsnr_ssim, t1w_lowsnr_ssimmap] = ssim(t1w_lowres_norm,t1w_norm);
[pred_ssim, pred_ssimmap] = ssim(pred_norm,t1w_norm);

%% plot

figure('Position',[200, 200, 800, 400]);
subplot(1,2,1);
imshow(t1w_lowsnr_diff(:, :, 100), [-0.5, 0.5]);
title('t1w lowres diff');
xlabel(['MAD=',num2str(round(t1w_lowsnr_mad*1e4)/1e4),...
        ' PSNR=',num2str(round(t1w_lowsnr_psnr*1e4)/1e4)]);
subplot(1,2,2);
imshow(pred_diff(:, :, 100), [-0.5, 0.5]);
title('pred diff');
xlabel(['MAD=',num2str(round(pred_mad*1e4)/1e4),...
        ' PSNR=',num2str(round(pred_psnr*1e4)/1e4)]);
fpSave = fullfile(dpSave, 'diff.png');
saveas(gcf,fpSave);

figure('Position',[200, 200, 800, 400]); 
subplot(1,2,1);
imshow(t1w_lowsnr_ssimmap(:, :, 100), [0, 1]);
title('t1w lowres');
xlabel(['SSIM=',num2str(round(t1w_lowsnr_ssim*1e4)/1e4)]);
subplot(1,2,2);
imshow(pred_ssimmap(:, :, 100), [0, 1]);
title('pred');
xlabel(['SSIM=',num2str(round(pred_ssim*1e4)/1e4)]);
fpSave = fullfile(dpSave, 'SSIM.png');
saveas(gcf,fpSave);


function [img_norm, imgres_norm]=norm(img, imgres, mask)
% standardize and normalize image to the range of [0, 1]

img_mean = mean(img(mask), 'all');
img_std = std(img(mask),0,'all');

img_norm = (img - img_mean) ./ img_std .* mask;
imgres_norm = (imgres - img_mean) / img_std .* mask;

img_norm = (img_norm + 3)/6;
imgres_norm = (imgres_norm + 3)/6;

end
