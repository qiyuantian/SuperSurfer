# s_cnnApply.py
#
#
# QT 2022

# %% load modual
import os
import scipy.io as sio
import numpy as np
import nibabel as nb
import glob
from matplotlib import pyplot as plt

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import tensorflow as tf
import utils as utils
from supersurfer import supersurfer_3d_model

# %% set gpu
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_visible_devices(devices=gpus[0:2], device_type='GPU')

# %% set up path
dpRoot = os.path.dirname(os.path.abspath('s_SuperSurfer_apply.py'))
os.chdir(dpRoot)

dpData = os.path.join(dpRoot, 'data')
dpSim = os.path.join(dpRoot, 'data-sim')

# %% load model
dpCnn = os.path.join(dpRoot, 'supersurfer')
fnCp = 'supersurfer_ep60'
fpCp = os.path.join(dpCnn, fnCp + '.h5')

dtnet = load_model(fpCp, custom_objects={'mean_squared_error_weighted': utils.mean_squared_error_weighted})

# %% files
files = sorted(glob.glob(os.path.join(dpData, 'hcp*')))

# %%
for ii in np.arange(0, 1): # start from 1, use 0th subject for evaluation
    
    fnData = os.path.basename(files[ii])
    print(fnData)
    
    # %% load data
    fpData = os.path.join(dpData, fnData) 
    tmp = sio.loadmat(fpData)
    mask = tmp['mask']
    t1w = tmp['t1w']
    
    fpSim = os.path.join(dpSim, fnData[:-4] + '_sim.mat')
    tmp = sio.loadmat(fpSim)
    t1w_lowres = tmp['t1w_lowres']

    t1w_lowres = np.expand_dims(t1w_lowres, 3)
    t1w = np.expand_dims(t1w, 3)
    mask = np.expand_dims(mask, 3)
    
    # %% standardize image intensities
    img_in, tmp = utils.normalize_image(t1w_lowres, t1w_lowres, mask)
    img_t1w, tmp = utils.normalize_image(t1w, t1w, mask)
    
    # indices for blocks
    ind_block, ind_brain = utils.block_ind(mask, sz_block=96) # you can use larger sz_pad to extract more blocks for data augmentation
    #print(ind_block)
        
    # %% one brain volume is too big for GPU memory, extract mxmxm blocks, 
    img_in_block = utils.extract_block(img_in, ind_block)
    img_t1w_block = utils.extract_block(img_t1w, ind_block)
    mask_block = utils.extract_block(mask, ind_block) # only compute loss within brain mask
        
    # %% apply cnn
    img_pred_block = np.zeros(img_in_block.shape)
    for mm in np.arange(0, img_in_block.shape[0]):
        tmp = dtnet.predict([img_in_block[mm:mm+1, :, :, :, :], mask_block[mm:mm+1, :, :, :, :]]) 
        img_pred_block[mm:mm+1, :, :, :, :] = tmp[:, :, :, :, 0:1] + img_in_block[mm:mm+1, :, :, :, :]
    
    img_pred, tmp = utils.block2brain(img_pred_block, ind_block, mask) # blocks to brain volume
    img_pred_denorm = utils.denormalize_image(t1w_lowres, img_pred, mask) # transform to normal intensity range

    dpPred = os.path.join(dpRoot, 'pred-supersurfer')
    if not os.path.exists(dpPred):
        os.mkdir(dpPred)
        print('create directory')
        
    fpPred = os.path.join(dpPred, fnData[0:6] + '_supersurfer.mat')
    sio.savemat(fpPred, {'img_pred_denorm': img_pred_denorm})
    
    # %% check data
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(img_in_block[10, :, :, 40, 0], clim=(-2, 2.), cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(img_pred_block[10, :, :, 40, 0], clim=(-2., 2.), cmap='gray')
    plt.subplot(2, 2, 4)
    plt.imshow(img_t1w_block[10, :, :, 40, 0], clim=(-2, 2.), cmap='gray')
    plt.subplot(2, 2, 3)
    plt.imshow(img_t1w_block[10, :, :, 40, 0] - img_pred_block[10, :, :, 40, 0], clim=(-2, 2.), cmap='gray')
    plt.savefig(dpRoot+'/supersurfer_ep60.png')
    plt.show()
         
    

