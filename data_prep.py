import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import os
import shutil
import random
import sklearn

# Create temporary folder to store the train and validation set splits
def remove_temp_train_val():
    if os.path.isdir('data/temp_train_val_split'):
        shutil.rmtree('data/temp_train_val_split') # os.rmdir('data/temp_train_val_split/')

    os.makedirs('data/temp_train_val_split/train_image')
    os.makedirs('data/temp_train_val_split/train_mask')
    os.makedirs('data/temp_train_val_split/val_image')    
    os.makedirs('data/temp_train_val_split/val_mask')        
    
def create_temp_train_val():
    remove_temp_train_val() # Clear the folder first
    
    imgs = []
    masks = []
    for f in os.listdir('data/raw/train_image/'):
        imgs.append(f)

    # for f in os.listdir('data/raw/train_mask/'):
    #     masks.append(f)

    file_names = sklearn.utils.shuffle(imgs)

    # imgs, masks = sklearn.utils.shuffle(imgs, masks)
    # print(file_names)
    # for i,m in zip(imgs, masks):

    for i,f in enumerate(file_names):
        if i < int(len(file_names) * 4/5):
            shutil.copy(src="data/raw/train_image/" + f, dst='data/temp_train_val_split/train_image')
            shutil.copy(src="data/raw/train_mask/" + f, dst='data/temp_train_val_split/train_mask')
        else:
            shutil.copy(src="data/raw/train_image/"+f, dst='data/temp_train_val_split/val_image/')
            shutil.copy(src="data/raw/train_mask/"+f, dst="data/temp_train_val_split/val_mask/")
            
# TODO: play with+
def create_datagens(validation_split=0.0):
    data_gen_args = dict(
        rescale = 1.0 / 255,
        zoom_range=0.1,
        shear_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=15,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=validation_split # depends on implementation
    )

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    return image_datagen, mask_datagen

def create_generators(method="constant_val", seed=1, batchsize=2, save_gen_train=False, save_gen_val=False):
    if method == "constant_val": # Use same validation set each epoch
        image_datagen, mask_datagen = create_datagens()
        create_temp_train_val()
        constant_val=True
        img_directory="data/temp_train_val_split/"
    elif method == "tranformed_val": # Use different transformed validation set each epoch
        image_datagen, mask_datagen = create_datagens(0.2)
        constant_val=False
        img_directory="data/raw/"
    else:
        None
    
    image_generator = image_datagen.flow_from_directory(
        img_directory,
        classes=["train_image"],
        class_mode=None,
        color_mode="grayscale",
        target_size=(512,512),
        batch_size=batchsize,
#         save_to_dir="data/generator_images/image" if save_gen_train else save_to_dir=None,
        save_to_dir="data/generator_images/image" if save_gen_train else None,
        seed=seed,
        subset=None if constant_val else "training"
    )

    mask_generator = mask_datagen.flow_from_directory(
        img_directory,
        classes=["train_mask"],
        class_mode=None,
        color_mode="grayscale",
        target_size=(512,512),
        batch_size=batchsize,
        save_to_dir="data/generator_images/mask" if save_gen_train else None,
        seed=seed,
        subset=None if constant_val else "training"
    )

    if constant_val: # If using constant validation, create new datagens to rescale
        print("Creating val_image_datagen and val_mask_datagen")
        val_image_datagen = ImageDataGenerator(rescale=1.0/255)
        val_mask_datagen = ImageDataGenerator(rescale=1.0/255)
    else: # If using different validations, use same datagens 
        print("Pointing val_image_datagen and val_mask_datagen")
        val_image_datagen = image_datagen
        val_mask_datagen = mask_datagen

    val_image_generator = val_image_datagen.flow_from_directory(
        img_directory,
        classes=["val_image"] if constant_val else ["train_image"],
        class_mode=None,
        color_mode="grayscale",
        target_size=(512,512),
        batch_size=batchsize,
        save_to_dir="data/generator_images/val_image/" if save_gen_val else None,
        seed=seed,
        subset=None if constant_val else "validation"
    )

    val_mask_generator = val_mask_datagen.flow_from_directory(
        img_directory,
        classes=["val_mask"] if constant_val else ["train_mask"],
        class_mode=None,
        color_mode="grayscale",
        target_size=(512,512),
        batch_size=batchsize,
        save_to_dir="data/generator_images/val_mask/" if save_gen_val else None,
        seed=seed,
        subset=None if constant_val else "validation"
    )

    train_generator = zip(image_generator, mask_generator)
    val_generator = zip(val_image_generator, val_mask_generator)
    
    return train_generator, val_generator




# THIS IMPLEMENTATION DOESN'T SEEM RIGHT: augmenting validation data on each iteration changes validation data each iteration
# and means the validation and test sets come from different distributions!

# image_generator = image_datagen.flow_from_directory('data/raw/', 
#                                                     classes=['train_image'], 
#                                                     class_mode=None, 
#                                                     color_mode='grayscale',
#                                                     target_size=(512,512),
#                                                     batch_size=batchsize, 
# #                                                     save_to_dir="data/generator_images/image",
#                                                     seed=seed,
#                                                     subset="training")

# mask_generator = mask_datagen.flow_from_directory('data/raw/', 
#                                                   classes=['train_mask'], 
#                                                   class_mode=None, 
#                                                   color_mode='grayscale',
#                                                   target_size=(512,512),
#                                                   batch_size=batchsize, 
# #                                                   save_to_dir="data/generator_images/mask",
#                                                   seed=seed,
#                                                   subset="training")

# val_image_generator = image_datagen.flow_from_directory('data/raw/',
#                                                         classes=['train_image'], 
#                                                         class_mode=None, 
#                                                         color_mode='grayscale',
#                                                         target_size=(512,512),
#                                                         batch_size=batchsize, 
#                                                         save_to_dir="data/generator_images/val_image",
#                                                         seed=seed,
#                                                         subset="validation")

# val_mask_generator = image_datagen.flow_from_directory('data/raw/', # TODO: fix this!!
#                                                         classes=['train_mask'], 
#                                                         class_mode=None, 
#                                                         color_mode='grayscale',
#                                                         target_size=(512,512),
#                                                         batch_size=batchsize, 
#                                                         save_to_dir="data/generator_images/val_mask",
#                                                         seed=seed,
#                                                         subset="validation")

# train_generator = zip(image_generator, mask_generator)
# val_generator = zip(val_image_generator, val_mask_generator)