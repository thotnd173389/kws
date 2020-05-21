from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
from kws_streaming.layers.compat import tf
from kws_streaming.layers.modes import Modes
from kws_streaming.layers.svdf import Svdf
from kws_streaming.layers.speech_features import SpeechFeatures
from kws_streaming.layers.stream import Stream


import os.path
import pprint
import numpy as np
import tensorflow.compat.v1 as tf
from kws_streaming.models import utils
from DataSettings import DataSettings

from tensorflow.keras import backend as K


def keyword_marvin_v1(input_shape = (48000,), dropout = 0.5):
    
    
    X_input = tf.keras.Input(input_shape)
    
    X = SpeechFeatures(
        frame_size_ms = 30.0,
        frame_step_ms = 10.0,
        mel_num_bins = 80,
        dct_num_features = 40,
        mel_upper_edge_hertz = 7000)(X_input)
    
    
    X = Svdf(
        units1=128, memory_size = 8, units2=32,
        activation='relu',
        pad=1,
        name='svdf_1')(X)
    
    X = tf.keras.layers.Dropout(dropout)(X)
    
    X = Svdf(
        units1=96, memory_size = 8, units2=32,
        activation='relu',
        pad=1,
        name='svdf_2')(X)
        
    X = tf.keras.layers.Dropout(dropout)(X)
    
    X = Svdf(
        units1=96, memory_size = 8, units2=32,
        activation='relu',
        pad=1,
        name='svdf_3')(X)
        
    X = tf.keras.layers.Dropout(dropout)(X)
    
    X = Svdf(
        units1=32, memory_size = 32, units2=-1,
        activation='relu',
        pad=1,
        name='svdf_4')(X)
    
    X = tf.keras.layers.Dropout(dropout)(X)
    
    X = Svdf(
        units1=32, memory_size = 32, units2=-1,
        activation='relu',
        pad=1,
        name='svdf_5')(X)
    
    X = tf.keras.layers.Dropout(dropout)(X)
    
    X = tf.keras.layers.Dense(units=1, activation = 'sigmoid')(X)
    
    
    # Create model
    model = tf.keras.models.Model(inputs=X_input, outputs=X, name='keyword_marvin_v1')
    
    return model



def keyword_marvin_v2(input_shape = (48000,), dropout = 0.):
    
    
    X_input = tf.keras.Input(input_shape)
    
    X = SpeechFeatures(
        frame_size_ms = 30.0,
        frame_step_ms = 10.0,
        mel_num_bins = 80,
        dct_num_features = 40,
        mel_upper_edge_hertz = 7000)(X_input)
    
    
    X = Svdf(
        units1=128, memory_size = 8, units2=32, dropout=dropout,
        activation='relu',
        pad=1,
        name='svdf_1')(X)
    
    
    X = Svdf(
        units1=96, memory_size = 8, units2=32, dropout=dropout,
        activation='relu',
        pad=1,
        name='svdf_2')(X)
        
    
    X = Svdf(
        units1=96, memory_size = 8, units2=32, dropout=dropout,
        activation='relu',
        pad=1,
        name='svdf_3')(X)
        
    
    X = Svdf(
        units1=32, memory_size = 32, units2=-1, dropout=dropout,
        activation='relu',
        pad=1,
        name='svdf_4')(X)
    
    
    X = Svdf(
        units1=32, memory_size = 32, units2=-1, dropout=dropout,
        activation='relu',
        pad=1,
        name='svdf_5')(X)
    
    X = tf.keras.layers.Dropout(dropout)(X)
    
    
    X = tf.keras.layers.Dense(units=1, activation = 'sigmoid')(X)
    
    
    # Create model
    model = tf.keras.models.Model(inputs=X_input, outputs=X, name='keyword_marvin_v2')
    
    return model
