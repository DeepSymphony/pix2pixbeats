'''
Lists different models to be used throughout the project
'''
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Lambda
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2DTranspose, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D, UpSampling2D
import numpy as np 

def lightweight_discriminator():
    img_shape = (96, 10, 1)
    test = np.ones((96,84,1))
    test = np.reshape(test,(1,96,84,1))

    model = Sequential()
    model.add(Conv2D(64, kernel_size=[1,12], strides=[1,12], input_shape=img_shape, padding="VALID"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(64, kernel_size=[1,7], strides=[1,7],padding="VALID"))
    model.add(LeakyReLU(alpha=.2))
    model.add(Conv2D(64, kernel_size=[2,1], strides=[2,1], padding="VALID"))
    model.add(LeakyReLU(alpha=.2))
    model.add(Conv2D(64, kernel_size=[2,1], strides=[2,1], padding="VALID"))
    model.add(LeakyReLU(alpha=.2))
    model.add(Conv2D(128, kernel_size=[4,1], strides=[2,1], padding="VALID"))
    model.add(LeakyReLU(alpha=.2))
    model.add(Conv2D(256, kernel_size=[3,1], strides=[2,1], padding="VALID"))
    model.add(LeakyReLU(alpha=.2))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=.2))
    model.add(Dense(1))
    return model

def super_lightweight_discriminator():

    model = Sequential()
    model.add(Conv2D(32, kernel_size=[1,2], strides=[1,2], input_shape=img_shape, padding="VALID"))
    
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(32, kernel_size=[1,3], strides=[1,3],padding="VALID"))
    
    model.add(LeakyReLU(alpha=.2))
    model.add(Conv2D(32, kernel_size=[2,1], strides=[2,1], padding="VALID"))
    
    model.add(LeakyReLU(alpha=.2))
    model.add(Conv2D(32, kernel_size=[2,1], strides=[2,1], padding="VALID"))
    model.add(LeakyReLU(alpha=.2))
    model.add(Conv2D(64, kernel_size=[4,1], strides=[2,1], padding="VALID"))
    model.add(LeakyReLU(alpha=.2))
    model.add(Conv2D(128, kernel_size=[3,1], strides=[2,1], padding="VALID"))
    model.add(LeakyReLU(alpha=.2))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=.2))
    model.add(Dense(1))
    return model

def pix2pix_generator(input_shape):
    encoder = Input(shape = input_shape)
    print(encoder.shape)
    encoder = Conv2D(32, kernel_size=[1,2], strides=[1,2], padding="VALID")(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Activation(LeakyReLU(alpha=.2))(encoder)
    print("first conv")
    print(encoder.shape)
    encoder = Conv2D(32, kernel_size=[1,3], strides = [1,3], padding = "VALID")(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Activation(LeakyReLU(alpha=.2))(encoder)
    print("second conv")
    print(encoder.shape)
    encoder = Conv2D(32, kernel_size=[2,1], strides=[2,1], padding="VALID")(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Activation(LeakyReLU(alpha=.2))(encoder)
    print("third conv")
    print(encoder.shape)
    encoder = Conv2D(32, kernel_size=[2,1], strides=[2,1], padding="VALID")(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Activation(LeakyReLU(alpha=.2))(encoder)
    print("fourth conv")
    print(encoder.shape)
    encoder = Conv2D(64, kernel_size=[4,1], strides=[2,1], padding="VALID")(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Activation(LeakyReLU(alpha=.2))(encoder)
    print("fifth conv")
    print(encoder.shape)
    encoder = Conv2D(128, kernel_size=[3,1], strides=[2,1], padding="VALID")(encoder)
    encoder = BatchNormalization()(encoder)
    encoder = Activation(LeakyReLU(alpha=.2))(encoder)
    print("sixth conv")
    print(encoder.shape)
    #now define the decoder
    decoder = Conv2DTranspose(64, (3,1), strides = (2,1))(encoder)
    decoder = BatchNormalization()(decoder)
    decoder = Dropout(.5)(decoder)
    decoder = Activation('relu')(decoder)
    print("decoder first")
    print(decoder.shape)
    decoder = Conv2DTranspose(32, (4,1), strides = (2,1))(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = Dropout(.5)(decoder)
    decoder = Activation('relu')(decoder)
    print("decoder second")
    print(decoder.shape)
    decoder = Conv2DTranspose(32, (2,1), strides = (2,1))(decoder) 
    decoder = BatchNormalization()(decoder)
    decoder = Dropout(.5)(decoder)
    decoder = Activation('relu')(decoder)
    print("decoder third")
    print(decoder.shape)
    decoder = Conv2DTranspose(32, (2,1), strides = (2,1))(decoder) 
    decoder = BatchNormalization()(decoder)
    decoder = Dropout(.5)(decoder)
    decoder = Activation('relu')(decoder)
    print("decoder fourth")
    print(decoder.shape)
    # decoder = Conv2DTranspose(32, (1,3), strides = (1,3))(decoder)
    decoder = UpSampling2D(size=(1,5))(decoder)
    decoder = Conv2D(32, kernel_size = [1,5], strides = [1,1],padding='same')(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = Dropout(.5)(decoder)
    decoder = Activation('relu')(decoder)

    ##need to add the last conv to complete decoder
    print("decoder fifth")
    print(decoder.shape)
    decoder = UpSampling2D(size=(1,2))(decoder)
    decoder = Conv2D(1, kernel_size = [1,5], strides = [1,1], padding='same')(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = Dropout(.5)(decoder)
    decoder = Activation('relu')(decoder)
    #below might be an extra layer but following this repo https://github.com/williamFalcon/pix2pix-keras/blob/master/pix2pix/networks/generator.py
    decoder = Conv2D(1, kernel_size = [1,10], strides = [1,1], padding='same')(decoder)
    return decode




