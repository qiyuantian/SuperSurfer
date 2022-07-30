# s_cnnTrain.py
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
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

import utils as utils
from supersurfer import supersurfer_3d_model
import tensorflow as tf

# %% set gpu
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_visible_devices(devices=gpus[0:2], device_type='GPU')

# %% set up path
dpRoot = os.path.dirname(os.path.abspath('s_SuperSurfer_train.py'))
os.chdir(dpRoot)

dpData = os.path.join(dpRoot, 'data')
dpSim = os.path.join(dpRoot, 'data-sim')

# %% files
files = sorted(glob.glob(os.path.join(dpData, 'hcp*')))

# %% load data 
img_in_block_all = np.zeros(1)
img_out_block_all = np.zeros(1)
mask_block_all = np.zeros(1)

for ii in np.arange(1, len(files)):  # start from 1, use 0th subject for evaluation

    fnData = os.path.basename(files[ii])
    print(fnData)

    # %% load data
    fpData = os.path.join(dpData, fnData)
    tmp = sio.loadmat(fpData)
    t1w = tmp['t1w']
    mask = tmp['mask']

    fpSim = os.path.join(dpSim, fnData[:-4] + '_sim.mat')
    tmp = sio.loadmat(fpSim)
    t1w_lowres = tmp['t1w_lowres']

    t1w = np.expand_dims(t1w, 3)  # expand dimension
    t1w_lowres = np.expand_dims(t1w_lowres, 3)
    mask = np.expand_dims(mask, 3)

    # %% standardize image intensities
    img_in, img_out = utils.normalize_image(t1w_lowres, t1w, mask)

    # indices for blocks
    ind_block, ind_brain = utils.block_ind(mask,
                                           sz_block=96)  # you can use larger sz_pad to extract more blocks for data augmentation
    # print(ind_block)

    # %% one brain volume is too big for GPU memory, extract mxmxm blocks, 
    img_in_block = utils.extract_block(img_in, ind_block)
    img_out_block = utils.extract_block(img_out, ind_block)
    mask_block = utils.extract_block(mask, ind_block)  # only compute loss within brain mask

    # %% flip along anatomically left-right to augment data
    tmp = np.flip(img_in_block, 1)
    img_in_block = np.concatenate((img_in_block, tmp), axis=0)

    tmp = np.flip(img_out_block, 1)
    img_out_block = np.concatenate((img_out_block, tmp), axis=0)

    tmp = np.flip(mask_block, 1)
    mask_block = np.concatenate((mask_block, tmp), axis=0)

    # %% concat all blocks together
    if img_out_block_all.any():
        img_in_block_all = np.concatenate((img_in_block_all, img_in_block), axis=0)
        img_out_block_all = np.concatenate((img_out_block_all, img_out_block), axis=0)
        mask_block_all = np.concatenate((mask_block_all, mask_block), axis=0)
    else:
        img_in_block_all = img_in_block
        img_out_block_all = img_out_block
        mask_block_all = mask_block

# %% compute residuals
img_res_block_all = img_out_block_all - img_in_block_all
# concat brain mask to specify regions to compute loss
img_res_block_all = np.concatenate((img_res_block_all, mask_block_all), axis=-1)

# %% check data
plt.imshow(img_in_block_all[10, :, :, 40, 0], clim=(-2, 2.), cmap='gray')
plt.imshow(img_out_block_all[10, :, :, 40, 0], clim=(-2., 2.), cmap='gray')
plt.imshow(img_res_block_all[10, :, :, 40, 0], clim=(-2., 2.), cmap='gray')
plt.imshow(img_res_block_all[10, :, :, 40, 1], clim=(-2., 2.), cmap='gray')
plt.imshow(mask_block_all[10, :, :, 40, 0], clim=(-2., 2.), cmap='gray')

# %% set up model
nfilter = 64
nin = 1
nout = 1
dtnet = supersurfer_3d_model(nin, nout, filter_num=nfilter)
dtnet.summary()

# %% set up adam
adam_opt = Adam(learning_rate=0.001)
dtnet.compile(loss=utils.mean_squared_error_weighted, optimizer=adam_opt)

# %%
dpCnn = os.path.join(dpRoot, 'supersurfer')
if not os.path.exists(dpCnn):
    os.mkdir(dpCnn)
    print('create directory')

# %% train cnn
nbatch = 1
nep = 60


# %% schedule lr
def scheduler(epoch, lr):
    if epoch < 15:
        return 0.001
    elif epoch < 35:
        return 0.0003
    else:
        return 0.0001

callback = LearningRateScheduler(scheduler, verbose=1)

# %% save best model
fnCp = 'supersurfer_ep' + str(nep)
fpCp = os.path.join(dpRoot, dpCnn, fnCp + '.h5')
checkpoint = ModelCheckpoint(fpCp, monitor='val_loss', save_best_only=True)

history = dtnet.fit(x=[img_in_block_all, mask_block_all],
                    y=img_res_block_all,
                    batch_size=nbatch,
                    validation_split=0.2,  # 20% samples for validation, 80% for training
                    epochs=nep,
                    callbacks=[callback, checkpoint],
                    shuffle=True,
                    verbose=1)

# save loss
fpLoss = os.path.join(dpRoot, dpCnn, fnCp + '.mat')
sio.savemat(fpLoss, {'loss_train': history.history['loss'], 'loss_val': history.history['val_loss']})
