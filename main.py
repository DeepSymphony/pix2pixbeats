'''
Map simplified beats to complex beats using pix2pix with midi data as images
'''
import numpy as np
import os

from keras.optimizers import Adam
from utils.facades_generator import facades_generator
from networks.generator import UNETGenerator
from networks.discriminator import PatchGanDiscriminator
from networks.DCGAN import DCGAN
from utils import patch_utils
from utils import logger
import time

from keras.utils import generic_utils as keras_generic_utils

from models import pix2pix_generator, super_lightweight_discriminator


# width, height of images to work with. Assumes images are square
im_width = 96
im_height = 10

# inpu/oputputt channels in image
input_channels = 1
output_channels = 1

# image dims
input_img_dim = (input_channels, im_width, im_height)
output_img_dim = (output_channels, im_width, im_height)

generator = pix2pix_generator(input_img_dim)

discriminator = super_lightweight_discriminator()

opt_discriminator = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)


discriminator.compile(loss='binary_crossentropy', optimizer=opt_discriminator)

batch_size = 1
nb_epoch = 100
n_images_per_epoch = 10

print('Training starting...')
for epoch in range(0, nb_epoch):
    print('Epoch {}'.format(epoch))
    batch_counter = 1
    start = time.time()
    progbar = keras_generic_utils.Progbar(n_images_per_epoch)

    # go through 1... n_images_per_epoch (which will go through all buckets as well
    for mini_batch_i in range(0, n_images_per_epoch, batch_size)