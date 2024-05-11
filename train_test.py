# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("always")
warnings.filterwarnings("ignore")

import os
import math
import numpy as np
import pandas as pd
import glob
import random
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from IPython.display import clear_output
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
import shutil
import pickle
import sklearn
import tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img, load_img
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.layers import Activation, MaxPool2D, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras.utils import normalize
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from keras_unet_collection import models, losses

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

'''--------------------seed everythings - reproducibility--------------------'''
def seed_all(seed=7):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

seed_all()

# '''------------------split data into train and validation and test sets--------------------'''
img_dir = r'endovis18-processed/images/'
mask_dir = r'endovis18-processed/masks_4class/'

image_paths = sorted([os.path.join(img_dir, fname) for fname in os.listdir(img_dir)])
mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)])


train_temp_images, test_images, train_temp_masks, test_masks = train_test_split(image_paths, mask_paths, test_size=0.1)
train_images, valid_images, train_masks, valid_masks = train_test_split(train_temp_images, train_temp_masks, test_size=0.2)

print(f'#training samples: {np.shape(train_images)[0]}, #validation samples: {np.shape(valid_images)[0]}, #test samples: {np.shape(test_images)[0]}')

class DataGenerator(keras.utils.Sequence):
    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths, n_classes):  ## img_size is a tuple of (height, width)
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.n_classes= n_classes

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
############################### reading images #############################################         
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = cv2.imread(path, -1)
            img= img / np.max(img)
            x[j] = img  ## for RGB image
#             x[j] = np.expand_dims(img, -1)/np.max(img)   ## for Grayscale image
    
############################### reading masks #############################################    
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            mask = cv2.imread(path, -1)
#             mask= np.array(mask)
#             y[j] = mask
            y[j] = np.expand_dims(mask, -1)
#         return x, y
        return x, tf.keras.utils.to_categorical(y, num_classes=self.n_classes).astype("float32")
    
# ### input img, mask sanity check ###
height=512
width= 512
num_class=4
batch_size=4
img_size= (height, width)
lr_rate= 1e-4
adam_opt = Adam(learning_rate = lr_rate, beta_1 = 0.9,beta_2 = 0.999)
epochs= 50
input_shape= (height, width, 3)

train_generator = DataGenerator(batch_size, img_size, train_images, train_masks, num_class)
valid_generator = DataGenerator(batch_size, img_size, valid_images, valid_masks, num_class)
test_generator = DataGenerator(batch_size, img_size, test_images, test_masks, num_class)


model = models.unet_2d(input_shape, 
                            filter_num=[64, 128, 256, 512, 1024],     # down- and upsampling levels. e.g., `[64, 128, 256, 512]`
                            n_labels=num_class,                               # multiclass segmentation
                            stack_num_down=2,                         #  number of convolutional layers per downsampling level/block
                            stack_num_up=2,                          # number of convolutional layers (after concatenation) per upsampling level/block.
                            activation='ReLU',                       #  one of the `tensorflow.keras.layers` or `keras_unet_collection.activations`
                            output_activation='Softmax',
                            batch_norm=True,                        # True for batch normalization
                            pool=False,                             # True or 'max' for MaxPooling2D --- 'ave' for AveragePooling2D --- False for strided conv + batch norm + activation
                            unpool=False,                           #  True or 'bilinear' for Upsampling2D with bilinear interpolation --- 'nearest' for Upsampling2D with nearest interpolation --- False for Conv2DTranspose + batch norm + activation
                            backbone=None,
                            weights=None,
                            name='simple_unet')


trained_model_dir = 'trained_model_dir/'
os.makedirs(trained_model_dir, exist_ok=True)

project_name= model.name+ '_endovis18_4class'
trained_model_filepath = os.path.join(trained_model_dir, project_name +'_model.h5')
trained_model_weights_filepath = os.path.join(trained_model_dir,'_model_weights.weights.h5')

train_steps=len(train_images)//batch_size
valid_steps= len(valid_masks)//batch_size

model.compile(optimizer = adam_opt, loss = losses.focal_tversky, metrics=['accuracy', losses.dice_coef])
history = model.fit(train_generator, 
                    epochs=epochs,
                    steps_per_epoch=train_steps,
                    validation_data=valid_generator, 
                    validation_steps=valid_steps,
                    )
test_predicted_mask_dir = trained_model_dir + 'prediction/'
os.makedirs(test_predicted_mask_dir, exist_ok=True)


def dice_coef(y_true, y_pred):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(y_true).astype(np.bool_)
    im2 = np.asarray(y_pred).astype(np.bool_)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())

def jaccard_index(A, B):
    '''Compute Jaccard index (IoU) between two segmentation masks.
    :param A: (numpy array) reference segmentaton mask
    :param B: (numpy array) predicted segmentaton mask
    :return: Jaccard index
    '''

    both = np.logical_and(A, B)
    either = np.logical_or(A, B)
    ji = int(np.sum(both)) / int(np.sum(either))

    return ji

scores = []
for i in range(len(test_images)):
    file_name= os.path.basename(test_images[i])
    test_img_path = test_images[i]
    ground_truth_path = test_masks[i]

    test_img= np.array(cv2.imread(test_img_path, -1))
    ground_truth= np.array(cv2.imread(ground_truth_path, -1))
    test_img= test_img / np.max(test_img)
    test_img_input=np.expand_dims(test_img, 0)
#     test_img_input=np.expand_dims(np.expand_dims(test_img, -1), 0) for grayscale images 
    print("test_img: {}, ground_truth:{}, test_img_input: {}".format(test_img.shape, ground_truth.shape, test_img_input.shape))
##     test_img: (512, 512), ground_truth:(512, 512), test_img_input: (1, 512, 512, 1)


    pred_proba = model.predict(test_img_input)
    pred = np.argmax(pred_proba, axis=3)
    pred = np.squeeze(pred, 0)
    print("pred prob shape: {}, pred argmax shape: {}, ground truth shape: {}".format(pred_proba.shape, pred.shape, ground_truth.shape))
##     pred prob shape: (1, 512, 512, 3), pred shape: (1, 512, 512)
    jaccard_score = jaccard_index(ground_truth, pred)
    dice_score = dice_coef(ground_truth, pred)
    print("{} -- image: {} -- jaccard index: {} -- dice score: {}".format(i, file_name, jaccard_score, dice_score))
    scores.append([file_name,jaccard_score, dice_score])
#     result_img= np.expand_dims(np.squeeze(pred, 0), -1)
    result_img= np.expand_dims(pred, -1)
    print(result_img.shape)
    result_img = array_to_img(result_img * 255 )
    result_img.save(os.path.join(test_predicted_mask_dir, file_name + '.png'))
    
    
df_result = pd.DataFrame(scores, columns=['file_name', 'jaccard_score (IoU)', 'dice_score'])
df_result.to_csv(os.path.join(trained_model_dir, project_name + '_jaccard_dice_scores_report.csv'), sep=',')
print("==========" *5)
print("Mean Dice: {}".format(np.mean(df_result['dice_score'].values)))
print("Mean IoU: {}".format(np.mean(df_result['jaccard_score (IoU)'].values)))
print("==========" *5)    