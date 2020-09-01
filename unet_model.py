import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, concatenate
from keras.models import Model

def unet_model(input_shape):
    """
    Set up the UNet model
    """
    X_input = Input(input_shape)

    # Begin Contracting Path
    conv1_a = Conv2D(64, (3,3), strides=(1,1), padding="same", name="conv1_a", activation="relu")(X_input) 
    conv1_b = Conv2D(64, (3,3), strides=(1,1), padding="same", name="conv1_b", activation="relu")(conv1_a)
    pool1_c = MaxPooling2D((2,2), name="pool1_c")(conv1_b)

    conv2_a = Conv2D(128, (3,3), strides=(1,1), padding="same", name="conv2_a", activation="relu")(pool1_c) 
    conv2_b = Conv2D(128, (3,3), strides=(1,1), padding="same", name="conv2_b", activation="relu")(conv2_a)
    pool2_c = MaxPooling2D((2,2), name="pool2_c")(conv2_b)

    conv3_a = Conv2D(256, (3, 3), activation="relu", padding="same")(pool2_c)
    conv3_b = Conv2D(256, (3, 3), activation="relu", padding="same")(conv3_a)
    pool3_c = MaxPooling2D((2, 2))(conv3_b)

    conv4_a = Conv2D(512, (3, 3), activation="relu", padding="same")(pool3_c)
    conv4_b = Conv2D(512, (3, 3), activation="relu", padding="same")(conv4_a)
    pool4_c = MaxPooling2D((2, 2))(conv4_b)
    
    conv3_a = Conv2D(256, (3,3), strides=(1,1), padding="same", name="conv3_a", activation="relu")(pool2_c) 
    conv3_b = Conv2D(256, (3,3), strides=(1,1), padding="same", name="conv3_b", activation="relu")(conv3_a)
    pool3_c = MaxPooling2D((2,2), name="pool3_c")(conv3_b)
    
    conv4_a = Conv2D(512, (3,3), strides=(1,1), padding="same", name="conv4_a", activation="relu")(pool3_c) 
    conv4_b = Conv2D(512, (3,3), strides=(1,1), padding="same", name="conv4_b", activation="relu")(conv4_a)
    pool4_c = MaxPooling2D((2,2), name="pool4_c")(conv4_b)    
    
    conv5_a = Conv2D(1024, (3,3), strides=(1,1), padding="same", name="conv5_a", activation="relu")(pool4_c)
    conv5_b = Conv2D(1024, (3,3), strides=(1,1), padding="same", name="conv5_b", activation="relu")(conv5_a)

    # Begin Expanding Path
    upconv6_a = Conv2DTranspose(512, (3,3), strides=(2,2), padding="same", activation="relu", name="upconv6_a")(conv5_b)  # TODO: Should I use an activation here?? 
    upconv6_b = concatenate([conv4_b, upconv6_a], axis=-1, name="upconv6_b")
    conv6_c = Conv2D(512, (3,3), strides=(1,1), padding="same", name="conv6_c", activation="relu")(upconv6_b)
    conv6_d = Conv2D(512, (3,3), strides=(1,1), padding="same", name="conv6_d", activation="relu")(conv6_c)
    
    upconv7_a = Conv2DTranspose(256, (3,3), strides=(2,2), padding="same", activation="relu", name="upconv7_a")(conv6_d) 
    upconv7_b = concatenate([conv3_b, upconv7_a], axis=-1, name="upconv7_b")
    conv7_c = Conv2D(256, (3,3), strides=(1,1), padding="same", name="conv7_c", activation="relu")(upconv7_b)
    conv7_d = Conv2D(256, (3,3), strides=(1,1), padding="same", name="conv7_d", activation="relu")(conv7_c)
    
    upconv8_a = Conv2DTranspose(128, (3,3), strides=(2,2), padding="same", activation="relu", name="upconv8_a")(conv7_d)
    upconv8_b = concatenate([conv2_b, upconv8_a], axis=-1, name="upconv8_b")
    conv8_c = Conv2D(128, (3,3), strides=(1,1), padding="same", name="conv8_c", activation="relu")(upconv8_b)
    conv8_d = Conv2D(128, (3,3), strides=(1,1), padding="same", name="conv8_d", activation="relu")(conv8_c)
    
    upconv9_a = Conv2DTranspose(64, (3,3), strides=(2,2), padding="same", activation="relu", name="upconv9_a")(conv8_d)
    upconv9_b = concatenate([conv1_b, upconv9_a], axis=-1, name="upconv9_b")
    conv9_c = Conv2D(64, (3,3), strides=(1,1), padding="same", name="conv9_c", activation="relu")(upconv9_b)
    conv9_d = Conv2D(64, (3,3), strides=(1,1), padding="same", name="conv9_d", activation="relu")(conv9_c)
    
#     conv10 = Conv2D(2, (1,1), strides=(1,1), name="conv10", activation="sigmoid")(conv9_d)
    conv10 = Conv2D(1, (1,1), strides=(1,1), name="conv10", activation="sigmoid")(conv9_d)

    unet_model = Model(inputs=X_input, outputs=conv10, name="UNet")

    return unet_model


# import tensorflow as tf
# from tensorflow import keras
# from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, concatenate
# from keras.models import Model
# import numpy as np
# import pandas as pd

# from namefornow import contract

# def unet_model(input_shape):
#     """
#     Set up the UNet model
#     """
#     X_input = Input(input_shape)

#     # Begin Contracting Path
#     conv1_a, conv1_b, pool1_c = contract(input=X_input, layer_num=1, num_channels=64, window=3, stride=1)
    
#     conv2_a, conv2_b, pool2_c = contract(input=pool1_c, layer_num=2, num_channels=128, window=3, stride=1)
    
#     conv3_a, conv3_b, pool3_c = contract(input=pool2_c, layer_num=3, num_channels=256, window=3, stride=1)
    
#     conv4_a, conv4_b, pool4_c = contract(input=pool3_c, layer_num=4, num_channels=512, window=3, stride=1)
    
#     conv5_a = Conv2D(1024, (3,3), strides=(1,1), name="conv5_a", activation="relu")(pool4_c)
#     conv5_b = Conv2D(1024, (3,3), strides=(1,1), name="conv5_b", activation="relu")(conv5_a)

#     # Begin Expanding Path
#     upconv6_a = Conv2DTranspose(512, (2,2), strides=(2,2), activation="relu", name="upconv6_a")(conv5_b)  # TODO: Should I use an activation here?? 
#     temp = int((conv4_b.shape[1] - upconv6_a.shape[1]) / 2)
#     temp2 = upconv6_a.shape[1] + temp
#     # print(temp)
#     # upconv6_b = concatenate([conv4_b[:,4:60,4:60,:], upconv6_a], axis=-1, name="upconv6_b")
#     upconv6_b = concatenate([conv4_b[:,temp:temp2,temp:temp2,:], upconv6_a], axis=-1, name="upconv6_b")
#     conv6_c = Conv2D(512, (3,3), strides=(1,1), name="conv6_c", activation="relu")(upconv6_b)
#     conv6_d = Conv2D(512, (3,3), strides=(1,1), name="conv6_d", activation="relu")(conv6_c)
    
#     upconv7_a = Conv2DTranspose(256, (2,2), strides=(2,2), activation="relu", name="upconv7_a")(conv6_d) 
#     temp = int((conv3_b.shape[1] - upconv7_a.shape[1]) / 2)
#     temp2 = upconv7_a.shape[1] + temp
#     # upconv7_b = concatenate([conv3_b[:,16:120,16:120,:], upconv7_a], axis=-1, name="upconv7_b")
#     upconv7_b = concatenate([conv3_b[:,temp:temp2,temp:temp2,:], upconv7_a], axis=-1, name="upconv7_b")
#     conv7_c = Conv2D(256, (3,3), strides=(1,1), name="conv7_c", activation="relu")(upconv7_b)
#     conv7_d = Conv2D(256, (3,3), strides=(1,1), name="conv7_d", activation="relu")(conv7_c)
    
#     upconv8_a = Conv2DTranspose(128, (2,2), strides=(2,2), activation="relu", name="upconv8_a")(conv7_d)
#     temp = int((conv2_b.shape[1] - upconv8_a.shape[1]) / 2)
#     temp2 = upconv8_a.shape[1] + temp
#     # upconv8_b = concatenate([conv2_b[:,40:240,40:240,:], upconv8_a], axis=-1, name="upconv8_b")
#     upconv8_b = concatenate([conv2_b[:,temp:temp2,temp:temp2,:], upconv8_a], axis=-1, name="upconv8_b")
#     conv8_c = Conv2D(128, (3,3), strides=(1,1), name="conv8_c", activation="relu")(upconv8_b)
#     conv8_d = Conv2D(128, (3,3), strides=(1,1), name="conv8_d", activation="relu")(conv8_c)
    
#     upconv9_a = Conv2DTranspose(64, (2,2), strides=(2,2), activation="relu", name="upconv9_a")(conv8_d)
#     temp = int((conv1_b.shape[1] - upconv9_a.shape[1]) / 2)
#     temp2 = upconv9_a.shape[1] + temp
#     # upconv9_b = concatenate([conv1_b[:,88:480,88:480,:], upconv9_a], axis=-1, name="upconv9_b")
#     upconv9_b = concatenate([conv1_b[:,temp:temp2,temp:temp2,:], upconv9_a], axis=-1, name="upconv9_b")
#     conv9_c = Conv2D(64, (3,3), strides=(1,1), name="conv9_c", activation="relu")(upconv9_b)
#     conv9_d = Conv2D(64, (3,3), strides=(1,1), name="conv9_d", activation="relu")(conv9_c)
    
#     conv10 = Conv2D(2, (1,1), strides=(1,1), name="conv10", activation="sigmoid")(conv9_d)
    
#     unet_model = Model(inputs=X_input, outputs=conv10, name="UNet")
#     return unet_model


#     # # Begin Contracting Path
#     # conv1 = Conv2D(64, (3,3), strides=(1,1), name="conv1", activation="relu")(X_input)
#     # conv2 = Conv2D(64, (3,3), strides=(1,1), name="conv2", activation="relu")(conv1)
#     # pool2 = MaxPooling2D((2,2), name="pool2")(conv2)
    
#     # conv3 = Conv2D(128, (3,3), strides=(1,1), name="conv3", activation="relu")(pool2)
#     # conv4 = Conv2D(128, (3,3), strides=(1,1), name="conv4", activation="relu")(conv3)
#     # pool4 = MaxPooling2D((2,2), name="pool4")(conv4)
    
#     # conv5 = Conv2D(256, (3,3), strides=(1,1), name="conv5", activation="relu")(pool4)
#     # conv6 = Conv2D(256, (3,3), strides=(1,1), name="conv6", activation="relu")(conv5)
#     # pool6 = MaxPooling2D((2,2), name="pool6")(conv6)
    
#     # conv7 = Conv2D(512, (3,3), strides=(1,1), name="conv7", activation="relu")(pool6)
#     # conv8 = Conv2D(512, (3,3), strides=(1,1), name="conv8", activation="relu")(conv7)
#     # pool8 = MaxPooling2D((2,2), name="pool8")(conv8)
    
#     # conv9 = Conv2D(1024, (3,3), strides=(1,1), name="conv9", activation="relu")(pool8)
#     # conv10 = Conv2D(1024, (3,3), strides=(1,1), name="conv10", activation="relu")(conv9)

#     # # Begin Expanding Path
#     # upconv11 = Conv2DTranspose(512, (2,2), strides=(2,2), activation="relu")(conv10)  # TODO: Should I use an activation here??
#     # upconv11 = concatenate([conv8[:,4:60,4:60,:], upconv11], axis=-1, name="upconv11")
#     # conv12 = Conv2D(512, (3,3), strides=(1,1), name="conv12", activation="relu")(upconv11)
#     # conv13 = Conv2D(512, (3,3), strides=(1,1), name="conv13", activation="relu")(conv12)
    
#     # upconv14 = Conv2DTranspose(256, (2,2), strides=(2,2), activation="relu")(conv13)
#     # upconv14 = concatenate([conv6[:,16:120,16:120,:], upconv14], axis=-1, name="upconv14")
#     # conv15 = Conv2D(256, (3,3), strides=(1,1), name="conv14", activation="relu")(upconv14)
#     # conv16 = Conv2D(256, (3,3), strides=(1,1), name="conv15", activation="relu")(conv15)
    
#     # upconv17 = Conv2DTranspose(128, (2,2), strides=(2,2), activation="relu")(conv16)
#     # upconv17 = concatenate([conv4[:,40:240,40:240,:], upconv17], axis=-1, name="upconv17")
#     # conv18 = Conv2D(128, (3,3), strides=(1,1), name="conv18", activation="relu")(upconv17)
#     # conv19 = Conv2D(128, (3,3), strides=(1,1), name="conv19", activation="relu")(conv18)
    
#     # upconv20 = Conv2DTranspose(64, (2,2), strides=(2,2), activation="relu")(conv19)
#     # upconv20 = concatenate([conv2[:,88:480,88:480,:], upconv20], axis=-1, name="upconv20")
#     # conv21 = Conv2D(64, (3,3), strides=(1,1), name="conv21", activation="relu")(upconv20)
#     # conv22 = Conv2D(64, (3,3), strides=(1,1), name="conv22", activation="relu")(conv21)
    
#     # conv23 = Conv2D(2, (1,1), strides=(1,1), name="conv23", activation="sigmoid")(conv22)
    
#     # unet_model = Model(inputs=X_input, outputs=conv23, name="UNet")
#     # return unet_model