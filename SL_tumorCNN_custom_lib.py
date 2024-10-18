## common libraries used in preprocessing, training and testing
import pickle
import numpy as np
from keras import backend as K
from keras.utils.np_utils import to_categorical
import os
from os import walk
import nibabel as nib
from numpy.lib import stride_tricks
import tensorflow as tf
import random

def make_list_of_files(path):
    file_list = []
    for (dirpath, dirnames, filenames) in walk(path):
        file_list.extend(filenames)
        break

    return file_list

def f1_score(y_true, y_pred):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection ) / (K.sum(y_true_f) + K.sum(y_pred_f) +  K.epsilon())
    return dice

def f1_score_3ch(y_true, y_pred):
    # y_true = K.cast(y_true, dtype='float32')
    # y_pred = K.cast(y_pred, dtype='float32')
    y_true = y_true[...,1:]
    y_pred = y_pred[...,1:]

    f1_score_val = f1_score(y_true, y_pred)

    return f1_score_val

def f1_score_3ch_whole(y_true, y_pred):

    y_pred = K.one_hot(K.argmax(y_pred, axis=-1), 4)

    y_true = y_true[...,2:]
    y_pred = y_pred[...,2:]

    f1_score_val = f1_score(y_true, y_pred)

    return f1_score_val

def f1_score_3ch_core(y_true, y_pred):

    y_pred = K.one_hot(K.argmax(y_pred, axis=-1), 4)
    y_true = y_true[...,2]
    y_pred = y_pred[...,2]


    f1_score_val = f1_score(y_true, y_pred)

    return f1_score_val

def f1_score_3ch_enh(y_true, y_pred):

    y_pred = K.one_hot(K.argmax(y_pred, axis=-1), 4)
    y_true = y_true[...,3]
    y_pred = y_pred[...,3]


    f1_score_val = f1_score(y_true, y_pred)

    return f1_score_val


def generate_UM_filenames(source_dir):
    source_files_unsorted = []

    # Iterate directory
    for path in os.listdir(source_dir):
        # check if current path is a file
        if os.path.isfile(os.path.join(source_dir, path)):
            source_files_unsorted.append(path)


    source_files = ['']*5

    #after getting the file list, sort the files by data
    #fspgr as first element, and so on


    for item in range(5):

        if  "C" not in source_files_unsorted[item] and "FSPGR" in source_files_unsorted[item]:
            source_files[0] = source_files_unsorted[item]

        if  "C" in source_files_unsorted[item] and "FSPGR" in source_files_unsorted[item]:
            source_files[1] = source_files_unsorted[item]


        if  "T2" in source_files_unsorted[item] and "FLAIR" in source_files_unsorted[item]:
            source_files[2] = source_files_unsorted[item]


        if  "T2" in source_files_unsorted[item] and "FSE" in source_files_unsorted[item]:
            source_files[3] = source_files_unsorted[item]

        if  "BRATUMIA" in source_files_unsorted[item]:
                source_files[4] = source_files_unsorted[item]


    del(source_files_unsorted)
    return source_files


def generate_BRATS_filenames(dataset_num):

    source_files = ['']*5
    source_file_keywords = ['FSPGR', 'FSPGR_C', 'FLAIR', 'T2FSE', 'label']
    for item in range(5):
        source_files[item] = f"rwBRATS_{dataset_num}_{source_file_keywords[item]}.nii"
    return source_files


def import_files(source_dir,source_files, dataset_group):


    for item in source_files:
        source_slice = nib.load(f"{source_dir}{item}")
        source_slice = source_slice.get_fdata()


        if item == source_files[0]:
            img_size = (5, source_slice.shape[0], source_slice.shape[1], source_slice.shape[2])
            source = np.zeros(img_size)


        if item == source_files[4]:

            source_slice_normalized = np.zeros(source_slice.shape)
            if dataset_group == 'BRATS':

                label_limits = [[-0.5,0.125],[0.125,0.375],[0.375,0.625],[0.625,1.5]] #BRATS
                for label in range(4):
                    source_slice_normalized= np.where(
                        (source_slice > label_limits[label][0]) &
                        (source_slice < label_limits[label][1]),
                        label, source_slice_normalized)


            elif dataset_group == 'UM':
                source_slice_normalized[ source_slice== 5] = 1
                source_slice_normalized[ source_slice== 6] = 2
                source_slice_normalized[ source_slice== 4] = 2
                source_slice_normalized[ source_slice== 7] = 3

            #case statement here edema 5 =>1, nonenh = 6,4 => 2, enh = 7 => 3


            source_slice = source_slice_normalized




        source[source_files.index(item),:,:,:] = source_slice


    source[np.isnan(source)] = 0
    return source


def cutup(data, blck, strd):
    sh = np.array(data.shape)
    blck = np.asanyarray(blck)
    strd = np.asanyarray(strd)
    nbl = (sh - blck) // strd + 1
    strides = np.r_[data.strides * strd, data.strides]
    dims = np.r_[nbl, blck]
    data6 = stride_tricks.as_strided(data, strides=strides, shape=dims)
    return data6#.reshape(-1, *blck)


def uncutup(strd,patches,blck_size,orig_shape):

    data = np.zeros(orig_shape)

    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            for k in range(patches.shape[2]):

                data[

                    i*strd[0]:blck_size[0]+i*strd[0],
                    j*strd[1]:blck_size[1]+j*strd[1],
                    k*strd[2]:blck_size[2]+k*strd[2]

                    ] = patches[i,j,k,:,:,:]
    return data



def load_img(file_list,parentdir,foldername,aug_p):

    for i, patch_name in enumerate(file_list):

        with open(f'{parentdir}\\{foldername}\\{patch_name}', 'rb') as f:
            training_testing_data = pickle.load(f)
        [
            X_train,
            y_train,
        ] = training_testing_data[0:2]
        ## data augmentation goes here
        if random.random() < aug_p:
            X_train = np.flip(X_train, 1)
            y_train = np.flip(y_train, 1)

        if i == 0:
            X = X_train
            y = y_train
        else:
            X = np.append(X,X_train,axis=0)
            y = np.append(y,y_train,axis=0)


    return(X,y)

def batch_maker_debug(batch_size,file_list,parentdir,foldername,num_classes=4,num_in_ch = 4,aug_p = 0):


    L = len(file_list)
    while True:
        batch_start = 0
        batch_end = batch_size


        while batch_start < L:
            limit = min(batch_end, L)

            X_batch,y_batch = load_img(file_list[batch_start:limit],parentdir,foldername,aug_p)
            y_batch = np.float32(y_batch) #remove when not needed

            if num_classes == 3:
                y_batch = y_batch[:,:,:,:,1:]
            if num_in_ch == 3:
                X_batch = X_batch[:,:,:,:,1:]

            yield (X_batch,y_batch,file_list[batch_start:limit]) #,file_list[batch_start:limit]

            batch_start += batch_size
            batch_end += batch_size

def batch_maker(batch_size,file_list,parentdir,foldername,num_classes=4,num_in_ch = 4,aug_p = 0):


    L = len(file_list)
    while True:
        batch_start = 0
        batch_end = batch_size


        while batch_start < L:
            limit = min(batch_end, L)

            X_batch,y_batch = load_img(file_list[batch_start:limit],parentdir,foldername,aug_p)
            y_batch = np.float32(y_batch) #remove when not needed

            if num_classes == 3:
                y_batch = y_batch[:,:,:,:,1:]

            if num_in_ch == 3:
                X_batch = X_batch[:,:,:,:,1:]

            yield (X_batch,y_batch) #,file_list[batch_start:limit]

            batch_start += batch_size
            batch_end += batch_size

def get_class_weights(train_file_list,train_data):
    import pandas as pd
    columns = ['0','1', '2', '3']
    df = pd.DataFrame(columns=columns)

    for img in range(len(train_file_list)):
        # print(img)
        _,y = next(train_data)
        temp_image = np.argmax(y, axis=4)
        val, counts = np.unique(temp_image, return_counts=True)
        zipped = zip(columns, counts)
        conts_dict = dict(zipped)
        df.loc[len(df)] = conts_dict

    df = df.replace(np.nan, 0)
    label_0 = df['0'].sum()
    label_1 = df['1'].sum()
    label_2 = df['2'].sum()
    label_3 = df['3'].sum()
    total_labels = label_0 + label_1 + label_2 + label_3
    n_classes = 4
    #Class weights calculation: n_samples / (n_classes * n_samples_for_class)
    wt0 = round((total_labels/(n_classes*label_0)), 2) #round to 2 decimals
    wt1 = round((total_labels/(n_classes*label_1)), 2)
    wt2 = round((total_labels/(n_classes*label_2)), 2)
    wt3 = round((total_labels/(n_classes*label_3)), 2)

    return wt0,wt1,wt2,wt3


def w_categorical_crossentropy(y_true, y_pred, weights): #y_true, y_pred size (1,128,128,128,ch)
    from itertools import product
    num_classes = len(weights) #ch

    final_mask = K.zeros_like(y_pred[..., 0]) # (1,128,128,128)
    # extracting variable
    y_pred_max = K.max(y_pred, axis=-1) # ch
    y_pred_max = K.expand_dims(y_pred_max, -1) # ch,1
    y_pred_max_mat = K.equal(y_pred, y_pred_max) # ch,1

    # weight calculation
    for c_p, c_t in product(range(num_classes), range(num_classes)): #iterating across the weight array, from 1 to 9
        # += 1 * ch
        final_mask += (K.cast(weights[c_t, c_p],K.floatx()) * K.cast(y_pred_max_mat[..., c_p] ,K.floatx())* K.cast(y_true[..., c_t],K.floatx()))
        # final mask = weight at class * bool * truth
    return K.categorical_crossentropy(y_pred, y_true) * final_mask

def w_cce_dice(y_true, y_pred, weights): #y_true, y_pred size (1,128,128,128,ch)
    from itertools import product
    num_classes = len(weights) #ch

    final_mask = K.zeros_like(y_pred[..., 0]) # (1,128,128,128)
    # extracting variable
    y_pred_max = K.max(y_pred, axis=-1) # ch
    y_pred_max = K.expand_dims(y_pred_max, -1) # ch,1
    y_pred_max_mat = K.equal(y_pred, y_pred_max) # ch,1

    # weight calculation
    for c_p, c_t in product(range(num_classes), range(num_classes)): #iterating across the weight array, from 1 to 9
        # += 1 * ch
        final_mask += (K.cast(weights[c_t, c_p],K.floatx()) * K.cast(y_pred_max_mat[..., c_p] ,K.floatx())* K.cast(y_true[..., c_t],K.floatx()))
        # final mask = weight at class * bool * truth

    intersection = K.sum(y_true * y_pred)
    dice = (2. * intersection) / (K.sum(y_true) + K.sum(y_pred)+ K.epsilon())

    dice_loss =  - K.log(dice + K.epsilon())

    return (K.categorical_crossentropy(y_pred, y_true)  +dice_loss)* final_mask

def ws_cce_dice(y_true, y_pred, weights): #y_true, y_pred size (1,128,128,128,ch)

    num_classes = len(weights) #ch

    for item in range(num_classes):
        weight = K.cast(weights[item],K.floatx())
        intersection = K.sum(y_true * y_pred)
        dice = (2. * intersection) / (K.sum(y_true) + K.sum(y_pred)+ K.epsilon())
        dice_loss =  - K.log(dice + K.epsilon())
        weighted_loss = (K.categorical_crossentropy(y_pred, y_true)  +dice_loss)*weight
        sum_weighted_loss = K.sum(weighted_loss)

    return sum_weighted_loss

import darr
def make_pred_patches(parentdir,testdir,dataset_list,model,modelname):

  for dataset_num in dataset_list:
    dataset_num = "{:0>3d}".format(dataset_num)
    with open(f'{testdir}BRATS_test_patches_{dataset_num}.pkl', 'rb') as f: #change this
        test = pickle.load(f)


    [
      X_test,
      y_test,
      ] = test
    del(test)

    y_pred_argmax = np.zeros(y_test.shape[:-1])

    for i in range(X_test.shape[1]):
      for j in range(X_test.shape[2]):
        for k in range(X_test.shape[3]):

          y_pred_patch=model.predict(X_test[:,i,j,k,:,:,:,:])
          y_pred_patch=np.argmax(y_pred_patch, axis=4)
          y_pred_argmax[i,j,k,:,:,:] = y_pred_patch


    #argmax and save data
    y_test_argmax = np.argmax(y_test, axis=-1) #changed from 6
    y_test_argmax = np.expand_dims(y_test_argmax, axis=0)
    y_pred_argmax = np.expand_dims(y_pred_argmax, axis=0)
    data = np.concatenate((y_pred_argmax, y_test_argmax), axis=0)
    darr.asarray(f'{parentdir}BRATS_pred_{dataset_num}{modelname}.darr', data)