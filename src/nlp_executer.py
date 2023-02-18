import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras import layers
from official.nlp import optimization  # to create AdamW optimizer

import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)


import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32
seed = 42

#url = 'https://drive.google.com/file/d/1ldHGaMOxB_-HhTYooWgJaJY-571Rp9sv/view?usp=share_link'
#url = 'https://github.com/flo-weiss/twintransformationnlp/blob/flo/bertintegration/resources/dt2.tar.xz'


dataset = tf.keras.utils.get_file('dt2.tar.xz', origin=url,
                                  untar=True, cache_dir='.',
                                  cache_subdir='')

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'dt/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)

class_names = raw_train_ds.class_names
train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

val_ds = tf.keras.utils.text_dataset_from_directory(
    'dt/train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)

val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

test_ds = tf.keras.utils.text_dataset_from_directory(
    'dt/test',
    batch_size=batch_size)

test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)