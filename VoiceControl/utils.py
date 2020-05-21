#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 20:38:09 2020

@author: thorius
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
from kws_streaming.layers.compat import tf
from kws_streaming.layers.modes import Modes
from kws_streaming.layers import svdf
from kws_streaming.layers import speech_features
from kws_streaming.layers.stream import Stream


import os.path
import pprint
import numpy as np
import tensorflow.compat.v1 as tf
from kws_streaming.models import utils
from DataSettings import DataSettings

from tensorflow.keras import backend as K



def E2E_1stage_v1(input_shape=(16000,), data_settings = None, dropout = 0.):
    data_settings.window_size_ms = 30.0
    data_settings.window_stride_ms = 10.0
    data_settings.dct_num_features = 40
    data_settings.mel_num_bins = 80
    data_settings.mel_upper_edge_hertz = 7000
    
    X_input = tf.keras.Input(input_shape)
    X =  speech_features.SpeechFeatures(
        frame_size_ms = data_settings.window_size_ms,
        frame_step_ms = data_settings.window_stride_ms,
        mel_num_bins = data_settings.mel_num_bins,
        dct_num_features = data_settings.dct_num_features,
        mel_upper_edge_hertz = data_settings.mel_upper_edge_hertz)(X_input)
    
    X = svdf.Svdf(
        units1=128, memory_size = 8, units2=64, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_1')(X)

    X = svdf.Svdf(
        units1=128, memory_size = 8, units2=64, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_2')(X)
    X = svdf.Svdf(
        units1=128, memory_size = 8, units2=64, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_3')(X)
    X = svdf.Svdf(
        units1=128, memory_size = 8, units2=64, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_4')(X)
    X = svdf.Svdf(
        units1=64, memory_size = 64, units2=-1, dropout=dropout,
        activation='relu',
        pad=1,
        name='svdf_5')(X)
    X = svdf.Svdf(
        units1=64, memory_size = 64, units2=-1, dropout=dropout,
        activation='relu',
        pad=1,
        name='svdf_6')(X)
    X = svdf.Svdf(
        units1=64, memory_size = 64, units2=-1, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_7')(X)

    X = Stream(cell=tf.keras.layers.Flatten())(X)
    X = tf.keras.layers.Dropout(dropout)(X)
    X = tf.keras.layers.Dense(units=data_settings.label_count)(X)
    

    # Create model
    model = tf.keras.models.Model(inputs=X_input, outputs=X, name='E2E_1stage_v1')

    return model
    
def E2E_1stage_v2(input_shape=(16000,), data_settings = None, dropout = 0.2):
    X_input = tf.keras.Input(input_shape)
    X =  speech_features.SpeechFeatures(
        frame_size_ms = data_settings.window_size_ms,
        frame_step_ms = data_settings.window_stride_ms)(X_input)
    
    X = svdf.Svdf(
        units1=256, memory_size = 8, units2=64, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_1')(X)

    X = svdf.Svdf(
        units1=256, memory_size = 10, units2=64, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_2')(X)
    X = svdf.Svdf(
        units1=256, memory_size = 10, units2=128, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_3')(X)
    X = svdf.Svdf(
        units1=256, memory_size = 10, units2=128, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_4')(X)
    X = svdf.Svdf(
        units1=256, memory_size = 10, units2=128, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_5')(X)
    X = svdf.Svdf(
        units1=256, memory_size = 10, units2=-1, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_6')(X)

    X = Stream(cell=tf.keras.layers.Flatten())(X)
    X = tf.keras.layers.Dropout(dropout)(X)
    X = tf.keras.layers.Dense(units=data_settings.label_count)(X)
    

    # Create model
    model = tf.keras.models.Model(inputs=X_input, outputs=X, name='E2E_1stage_v2')

    return model
    
def E2E_1stage_v3(input_shape=(16000,), data_settings = None, dropout = 0.5):
    data_settings.window_size_ms = 40.0
    data_settings.window_stride_ms = 20.0
    data_settings.dct_num_features = 40
    data_settings.mel_num_bins = 80
    data_settings.mel_upper_edge_hertz = 7000
    
    X_input = tf.keras.Input(input_shape)
    X =  speech_features.SpeechFeatures(
        frame_size_ms = data_settings.window_size_ms,
        frame_step_ms = data_settings.window_stride_ms,
        mel_num_bins = data_settings.mel_num_bins,
        dct_num_features = data_settings.dct_num_features,
        mel_upper_edge_hertz = data_settings.mel_upper_edge_hertz)(X_input)
    
    X = svdf.Svdf(
        units1=256, memory_size = 10, units2=64, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_1')(X)

    X = svdf.Svdf(
        units1=256, memory_size = 10, units2=64, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_2')(X)
    X = svdf.Svdf(
        units1=256, memory_size = 10, units2=64, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_3')(X)
    X = svdf.Svdf(
        units1=256, memory_size = 10, units2=64, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_4')(X)
    X = svdf.Svdf(
        units1=64, memory_size = 10, units2=-1, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_5')(X)

    X = Stream(cell=tf.keras.layers.Flatten())(X)
    X = tf.keras.layers.Dropout(dropout)(X)
    X = tf.keras.layers.Dense(units=data_settings.label_count)(X)
    

    # Create model
    model = tf.keras.models.Model(inputs=X_input, outputs=X, name='E2E_1stage_v3')

    return model

def E2E_1stage_v4(input_shape=(16000,), data_settings = None, dropout = 0.5):
    data_settings.window_size_ms = 40.0
    data_settings.window_stride_ms = 20.0
    data_settings.dct_num_features = 40
    data_settings.mel_num_bins = 80
    data_settings.mel_upper_edge_hertz = 7000
    
    X_input = tf.keras.Input(input_shape)
    X =  speech_features.SpeechFeatures(
        frame_size_ms = data_settings.window_size_ms,
        frame_step_ms = data_settings.window_stride_ms,
        mel_num_bins = data_settings.mel_num_bins,
        dct_num_features = data_settings.dct_num_features,
        mel_upper_edge_hertz = data_settings.mel_upper_edge_hertz)(X_input)
    
    X = svdf.Svdf(
        units1=256, memory_size = 4, units2=128, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_1')(X)

    X = svdf.Svdf(
        units1=256, memory_size = 12, units2=128, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_2')(X)
    X = svdf.Svdf(
        units1=256, memory_size = 12, units2=128, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_3')(X)
    X = svdf.Svdf(
        units1=256, memory_size = 12, units2=128, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_4')(X)
    X = svdf.Svdf(
        units1=256, memory_size = 12, units2=-1, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_5')(X)

    X = Stream(cell=tf.keras.layers.Flatten())(X)
    X = tf.keras.layers.Dropout(dropout)(X)
    X = tf.keras.layers.Dense(units=data_settings.label_count)(X)
    

    # Create model
    model = tf.keras.models.Model(inputs=X_input, outputs=X, name='E2E_1stage_v4')

    return model
    
    
def E2E_1stage_v5(input_shape=(16000,), data_settings = None, dropout = 0.5):
    data_settings.window_size_ms = 30.0
    data_settings.window_stride_ms = 10.0
    data_settings.dct_num_features = 40
    data_settings.mel_num_bins = 80
    data_settings.mel_upper_edge_hertz = 7000
    
    X_input = tf.keras.Input(input_shape)
    X =  speech_features.SpeechFeatures(
        frame_size_ms = data_settings.window_size_ms,
        frame_step_ms = data_settings.window_stride_ms,
        mel_num_bins = data_settings.mel_num_bins,
        dct_num_features = data_settings.dct_num_features,
        mel_upper_edge_hertz = data_settings.mel_upper_edge_hertz)(X_input)
    
    X = svdf.Svdf(
        units1=256, memory_size = 8, units2=64, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_1')(X)

    X = svdf.Svdf(
        units1=256, memory_size = 8, units2=64, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_2')(X)
    X = svdf.Svdf(
        units1=256, memory_size = 8, units2=64, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_3')(X)
    X = svdf.Svdf(
        units1=32, memory_size = 32, units2=-1, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_4')(X)
    X = svdf.Svdf(
        units1=32, memory_size = 32, units2=-1, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_5')(X)

    X = Stream(cell=tf.keras.layers.Flatten())(X)
    X = tf.keras.layers.Dropout(dropout)(X)
    X = tf.keras.layers.Dense(units=data_settings.label_count)(X)
    

    # Create model
    model = tf.keras.models.Model(inputs=X_input, outputs=X, name='E2E_1stage_v5')

    return model
    
def E2E_1stage_v6(input_shape=(16000,), data_settings = None, dropout = 0.5):
    data_settings.window_size_ms = 30.0
    data_settings.window_stride_ms = 10.0
    data_settings.dct_num_features = 40
    data_settings.mel_num_bins = 80
    data_settings.mel_upper_edge_hertz = 7000
    
    X_input = tf.keras.Input(input_shape)
    X =  speech_features.SpeechFeatures(
        frame_size_ms = data_settings.window_size_ms,
        frame_step_ms = data_settings.window_stride_ms,
        mel_num_bins = data_settings.mel_num_bins,
        dct_num_features = data_settings.dct_num_features,
        mel_upper_edge_hertz = data_settings.mel_upper_edge_hertz)(X_input)
    
    X = svdf.Svdf(
        units1=224, memory_size = 8, units2=96, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_1')(X)

    X = svdf.Svdf(
        units1=224, memory_size = 8, units2=96, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_2')(X)
    X = svdf.Svdf(
        units1=224, memory_size = 8, units2=96, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_3')(X)
    X = svdf.Svdf(
        units1=32, memory_size = 32, units2=-1, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_4')(X)
    X = svdf.Svdf(
        units1=32, memory_size = 32, units2=-1, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_5')(X)

    X = Stream(cell=tf.keras.layers.Flatten())(X)
    X = tf.keras.layers.Dropout(dropout)(X)
    X = tf.keras.layers.Dense(units=data_settings.label_count)(X)
    

    # Create model
    model = tf.keras.models.Model(inputs=X_input, outputs=X, name='E2E_1stage_v6')

    return model
    
def E2E_1stage_v7(input_shape=(16000,), data_settings = None, dropout = 0.5):
    data_settings.window_size_ms = 40.0
    data_settings.window_stride_ms = 20.0
    data_settings.dct_num_features = 40
    data_settings.mel_num_bins = 80
    data_settings.mel_upper_edge_hertz = 7000
    
    X_input = tf.keras.Input(input_shape)
    X =  speech_features.SpeechFeatures(
        frame_size_ms = data_settings.window_size_ms,
        frame_step_ms = data_settings.window_stride_ms,
        mel_num_bins = data_settings.mel_num_bins,
        dct_num_features = data_settings.dct_num_features,
        mel_upper_edge_hertz = data_settings.mel_upper_edge_hertz)(X_input)
    
    X = svdf.Svdf(
        units1=224, memory_size = 12, units2=56, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_1')(X)

    X = svdf.Svdf(
        units1=224, memory_size = 12, units2=56, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_2')(X)
    X = svdf.Svdf(
        units1=224, memory_size = 12, units2=56, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_3')(X)
    X = svdf.Svdf(
        units1=32, memory_size = 32, units2=-1, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_4')(X)
    X = svdf.Svdf(
        units1=32, memory_size = 32, units2=-1, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_5')(X)

    X = Stream(cell=tf.keras.layers.Flatten())(X)
    X = tf.keras.layers.Dropout(dropout)(X)
    X = tf.keras.layers.Dense(units=data_settings.label_count)(X)
    

    # Create model
    model = tf.keras.models.Model(inputs=X_input, outputs=X, name='E2E_1stage_v7')

    return model
    
    
def E2E_1stage_v8(input_shape=(16000,), data_settings = None, dropout = 0.):
    assert data_settings.wanted_words == 'on,off,up,down,zero,one,two,three,four,five,six,seven,eight,nine'
    assert data_settings.window_size_ms == 40.0
    assert data_settings.window_stride_ms == 20.0
    assert data_settings.dct_num_features == 40
    assert data_settings.mel_num_bins == 80
    assert data_settings.mel_upper_edge_hertz == 7000
    
    X_input = tf.keras.Input(input_shape)
    X =  speech_features.SpeechFeatures(
        frame_size_ms = data_settings.window_size_ms,
        frame_step_ms = data_settings.window_stride_ms,
        mel_num_bins = data_settings.mel_num_bins,
        dct_num_features = data_settings.dct_num_features,
        mel_upper_edge_hertz = data_settings.mel_upper_edge_hertz)(X_input)
    
    X = svdf.Svdf(
        units1=256, memory_size = 4, units2=128, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_1')(X)

    X = svdf.Svdf(
        units1=256, memory_size = 10, units2=128, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_2')(X)
    X = svdf.Svdf(
        units1=256, memory_size = 10, units2=128, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_3')(X)
    X = svdf.Svdf(
        units1=256, memory_size = 10, units2=128, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_4')(X)
    X = svdf.Svdf(
        units1=256, memory_size = 10, units2=128, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_5')(X)

    X = svdf.Svdf(
        units1=256, memory_size = 10, units2=-1, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_6')(X)


    X = Stream(cell=tf.keras.layers.Flatten())(X)
    X = tf.keras.layers.Dropout(dropout)(X)
    X = tf.keras.layers.Dense(units=data_settings.label_count)(X)
    

    # Create model
    model = tf.keras.models.Model(inputs=X_input, outputs=X, name='E2E_1stage_v8')

    return model

def E2E_1stage_v8_vl_0_3(input_shape=(16000,), data_settings = None, dropout = 0.):
    assert data_settings.wanted_words == 'on,off,up,down,zero,one,two,three,four,five,six,seven,eight,nine'
    assert data_settings.window_size_ms == 40.0
    assert data_settings.window_stride_ms == 20.0
    assert data_settings.dct_num_features == 40
    assert data_settings.mel_num_bins == 80
    assert data_settings.background_volume == 0.3
    assert data_settings.mel_upper_edge_hertz == 7000
    
    X_input = tf.keras.Input(input_shape)
    X =  speech_features.SpeechFeatures(
        frame_size_ms = data_settings.window_size_ms,
        frame_step_ms = data_settings.window_stride_ms,
        mel_num_bins = data_settings.mel_num_bins,
        dct_num_features = data_settings.dct_num_features,
        mel_upper_edge_hertz = data_settings.mel_upper_edge_hertz)(X_input)
    
    X = svdf.Svdf(
        units1=256, memory_size = 4, units2=128, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_1')(X)

    X = svdf.Svdf(
        units1=256, memory_size = 10, units2=128, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_2')(X)
    X = svdf.Svdf(
        units1=256, memory_size = 10, units2=128, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_3')(X)
    X = svdf.Svdf(
        units1=256, memory_size = 10, units2=128, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_4')(X)
    X = svdf.Svdf(
        units1=256, memory_size = 10, units2=128, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_5')(X)

    X = svdf.Svdf(
        units1=256, memory_size = 10, units2=-1, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_6')(X)


    X = Stream(cell=tf.keras.layers.Flatten())(X)
    X = tf.keras.layers.Dropout(dropout)(X)
    X = tf.keras.layers.Dense(units=data_settings.label_count)(X)
    

    # Create model
    model = tf.keras.models.Model(inputs=X_input, outputs=X, name='E2E_1stage_v8_vl_0_3')

    return model


def E2E_1stage_v8_vl_0_4(input_shape=(16000,), data_settings = None, dropout = 0.):
    assert data_settings.wanted_words == 'on,off,up,down,zero,one,two,three,four,five,six,seven,eight,nine'
    assert data_settings.window_size_ms == 40.0
    assert data_settings.window_stride_ms == 20.0
    assert data_settings.dct_num_features == 40
    assert data_settings.mel_num_bins == 80
    assert data_settings.background_volume == 0.4
    assert data_settings.mel_upper_edge_hertz == 7000
    
    X_input = tf.keras.Input(input_shape)
    X =  speech_features.SpeechFeatures(
        frame_size_ms = data_settings.window_size_ms,
        frame_step_ms = data_settings.window_stride_ms,
        mel_num_bins = data_settings.mel_num_bins,
        dct_num_features = data_settings.dct_num_features,
        mel_upper_edge_hertz = data_settings.mel_upper_edge_hertz)(X_input)
    
    X = svdf.Svdf(
        units1=256, memory_size = 4, units2=128, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_1')(X)

    X = svdf.Svdf(
        units1=256, memory_size = 10, units2=128, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_2')(X)
    X = svdf.Svdf(
        units1=256, memory_size = 10, units2=128, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_3')(X)
    X = svdf.Svdf(
        units1=256, memory_size = 10, units2=128, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_4')(X)
    X = svdf.Svdf(
        units1=256, memory_size = 10, units2=128, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_5')(X)

    X = svdf.Svdf(
        units1=256, memory_size = 10, units2=-1, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_6')(X)


    X = Stream(cell=tf.keras.layers.Flatten())(X)
    X = tf.keras.layers.Dropout(dropout)(X)
    X = tf.keras.layers.Dense(units=data_settings.label_count)(X)
    

    # Create model
    model = tf.keras.models.Model(inputs=X_input, outputs=X, name='E2E_1stage_v8_vl_0_4')

    return model

def E2E_1stage_v9(input_shape=(16000,), data_settings = None, dropout = 0.):
    assert data_settings.wanted_words == 'on,off,up,down,zero,one,two,three,four,five,six,seven,eight,nine'
    assert data_settings.window_size_ms == 40.0
    assert data_settings.window_stride_ms == 20.0
    assert data_settings.dct_num_features == 40
    assert data_settings.mel_num_bins == 80
    assert data_settings.mel_upper_edge_hertz == 7000
    
    X_input = tf.keras.Input(input_shape)
    X =  speech_features.SpeechFeatures(
        frame_size_ms = data_settings.window_size_ms,
        frame_step_ms = data_settings.window_stride_ms,
        mel_num_bins = data_settings.mel_num_bins,
        dct_num_features = data_settings.dct_num_features,
        mel_upper_edge_hertz = data_settings.mel_upper_edge_hertz)(X_input)
    
    X = svdf.Svdf(
        units1=192, memory_size = 4, units2=96, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_1')(X)

    X = svdf.Svdf(
        units1=192, memory_size = 10, units2=96, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_2')(X)
    X = svdf.Svdf(
        units1=192, memory_size = 10, units2=96, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_3')(X)
    X = svdf.Svdf(
        units1=192, memory_size = 10, units2=96, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_4')(X)
    X = svdf.Svdf(
        units1=192, memory_size = 10, units2=96, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_5')(X)

    X = svdf.Svdf(
        units1=192, memory_size = 10, units2=-1, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_6')(X)


    X = Stream(cell=tf.keras.layers.Flatten())(X)
    X = tf.keras.layers.Dropout(dropout)(X)
    X = tf.keras.layers.Dense(units=data_settings.label_count)(X)
    

    # Create model
    model = tf.keras.models.Model(inputs=X_input, outputs=X, name='E2E_1stage_v9')

    return model
    
def keyword_marvin_v1(input_shape=(16000,), data_settings = None, dropout = 0.5):
    
    assert data_settings.window_size_ms == 30.0
    assert data_settings.window_stride_ms == 10.0
    assert data_settings.dct_num_features == 40
    assert data_settings.mel_num_bins == 80
    assert data_settings.mel_upper_edge_hertz == 7000
    assert data_settings.wanted_words == 'marvin'
    
    X_input = tf.keras.Input(input_shape)
    X =  speech_features.SpeechFeatures(
        frame_size_ms = data_settings.window_size_ms,
        frame_step_ms = data_settings.window_stride_ms,
        mel_num_bins = data_settings.mel_num_bins,
        dct_num_features = data_settings.dct_num_features,
        mel_upper_edge_hertz = data_settings.mel_upper_edge_hertz)(X_input)
    
    X = svdf.Svdf(
        units1=96, memory_size = 8, units2=32, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_1')(X)

    X = svdf.Svdf(
        units1=96, memory_size = 8, units2=32, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_2')(X)
    X = svdf.Svdf(
        units1=96, memory_size = 8, units2=32, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_3')(X)
    X = svdf.Svdf(
        units1=32, memory_size = 32, units2=-1, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_4')(X)
    X = svdf.Svdf(
        units1=32, memory_size = 32, units2=-1, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_5')(X)

    X = Stream(cell=tf.keras.layers.Flatten())(X)
    X = tf.keras.layers.Dropout(dropout)(X)
    X = tf.keras.layers.Dense(units=data_settings.label_count)(X)
    

    # Create model
    model = tf.keras.models.Model(inputs=X_input, outputs=X, name='keyword_marvin_v1')

    return model


def keyword_marvin_v2(input_shape=(16000,), data_settings = None, dropout = 0.5):
    
    assert data_settings.window_size_ms == 30.0
    assert data_settings.window_stride_ms == 10.0
    assert data_settings.dct_num_features == 40
    assert data_settings.mel_num_bins == 80
    assert data_settings.mel_upper_edge_hertz == 7000
    assert data_settings.wanted_words == 'marvin'
    
    X_input = tf.keras.Input(input_shape)
    X =  speech_features.SpeechFeatures(
        frame_size_ms = data_settings.window_size_ms,
        frame_step_ms = data_settings.window_stride_ms,
        mel_num_bins = data_settings.mel_num_bins,
        dct_num_features = data_settings.dct_num_features,
        mel_upper_edge_hertz = data_settings.mel_upper_edge_hertz)(X_input)
    
    X = svdf.Svdf(
        units1=96, memory_size = 12, units2=32, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_1')(X)

    X = svdf.Svdf(
        units1=96, memory_size = 12, units2=32, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_2')(X)
    X = svdf.Svdf(
        units1=96, memory_size = 12, units2=32, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_3')(X)
    X = svdf.Svdf(
        units1=32, memory_size = 32, units2=-1, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_4')(X)
    X = svdf.Svdf(
        units1=32, memory_size = 32, units2=-1, dropout=dropout,
        activation='relu',
        pad=0,
        name='svdf_5')(X)

    X = Stream(cell=tf.keras.layers.Flatten())(X)
    X = tf.keras.layers.Dropout(dropout)(X)
    X = tf.keras.layers.Dense(units=data_settings.label_count)(X)
    

    # Create model
    model = tf.keras.models.Model(inputs=X_input, outputs=X, name='keyword_marvin_v2')

    return model
