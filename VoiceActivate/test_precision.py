#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 18:57:22 2020

@author: thorius
"""


# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import tensorflow.compat.v1 as tf
import input_data
import logging
import numpy as np
from kws_streaming.models import utils
from DataSettings import DataSettings
from TrainingSettings import TrainingSettings

from utils import keyword_marvin_v2
from utils import keyword_marvin_v3


# %%
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.info(tf.__version__)
logger.info("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

assert tf.__version__ == '2.1.0'

# %%
# Start a new TensorFlow session.
tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)



# %%
# setting data and training
data_settings = DataSettings(
    window_size_ms = 30.0,
    window_stride_ms = 10.0,
    dct_num_features = 40,
    mel_num_bins = 80,
    mel_upper_edge_hertz = 7000,
    silence_percentage = 100.0,
    unknown_percentage = 100.0,
    wanted_words = 'marvin')


training_settings = TrainingSettings()


# %%
time_shift_samples = int((data_settings.time_shift_ms * data_settings.sample_rate) / 1000)


# %%
audio_processor = input_data.AudioProcessor(data_settings)

# %%
# create model
model_non_stream_batch = keyword_marvin_v3(input_shape=(16000), data_settings=data_settings, dropout = 0.)

# load model's weights
weights_name = 'best_weights'
model_non_stream_batch.load_weights(os.path.join(training_settings.train_dir + model_non_stream_batch.name, weights_name))


test_fingerprints, test_ground_truth = audio_processor.get_data(
    -1, 0, data_settings, 0.0, 0.0, 0,
    'testing', 0.0, sess)

test_ground_pred = np.argmax(model_non_stream_batch.predict(test_fingerprints), axis = 1)

from sklearn.metrics import classification_report

print(classification_report(test_ground_truth, test_ground_pred))
