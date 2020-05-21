import math
import os
import input_data
from kws_streaming.models import utils

class DataSettings(object):
    def __init__(self, 
                 data_url = 'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
                 data_dir = './speech_data',
                 wanted_words = 'on,off,zero,one,two,three,four',
                 background_volume = 0.1,
                 background_frequency = 0.8,
                 split_data = 1, 
                 silence_percentage = 10.0, 
                 unknown_percentage = 10.0,
                 time_shift_ms = 100,
                 testing_percentage = 10, 
                 validation_percentage = 10,
                 sample_rate = 16000,
                 clip_duration_ms = 1000,
                 window_size_ms = 30.0,
                 window_stride_ms = 10.0,
                 preprocess = 'raw', 
                 feature_bin_count = 40,
                 summaries_dir = './summary/logs',
                 mel_upper_edge_hertz = 7000,
                 mel_num_bins=40,
                 dct_num_features=10,
                 batch_size = 100,
                 resample = 0.15):
        
        MS_PER_SECOND = 1000
        self.data_url = data_url
        self.data_dir = data_dir
        self.wanted_words = wanted_words
        self.background_volume = background_volume
        self.background_frequency = background_frequency
        self.split_data = split_data
        self.silence_percentage = silence_percentage
        self.unknown_percentage = unknown_percentage
        self.time_shift_ms = time_shift_ms
        self.testing_percentage = testing_percentage
        self.validation_percentage = validation_percentage
        self.sample_rate = sample_rate
        self.clip_duration_ms = clip_duration_ms
        self.window_stride_ms = window_stride_ms
        self.window_size_ms = window_size_ms
        self.preprocess = preprocess
        self.feature_bin_count = feature_bin_count
        self.summaries_dir= summaries_dir
        self.mel_upper_edge_hertz = mel_upper_edge_hertz
        self.mel_num_bins = mel_num_bins
        self.dct_num_features=dct_num_features
        self.batch_size = batch_size
        self.resample = resample

        # update data_settings
        
        label_count = len(
            input_data.prepare_words_list(
                self.wanted_words.split(','), self.split_data))
        desired_samples = int(self.sample_rate * self.clip_duration_ms /
                                MS_PER_SECOND)
        window_size_samples = int(self.sample_rate * self.window_size_ms /
                                    MS_PER_SECOND)
        window_stride_samples = int(self.sample_rate * self.window_stride_ms /
                                    MS_PER_SECOND)
        length_minus_window = (desired_samples - window_size_samples)
        if length_minus_window < 0:
            spectrogram_length = 0
        else:
            spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
        if self.preprocess == 'raw':
            average_window_width = -1
            fingerprint_width = desired_samples
            spectrogram_length = 1
        elif self.preprocess == 'average':
            fft_bin_count = 1 + (utils.next_power_of_two(window_size_samples) / 2)
            average_window_width = int(
                math.floor(fft_bin_count / self.feature_bin_count))
            fingerprint_width = int(
                math.ceil(float(fft_bin_count) / average_window_width))
        elif self.preprocess == 'mfcc':
            average_window_width = -1
            fingerprint_width = self.feature_bin_count
        elif self.preprocess == 'micro':
            average_window_width = -1
            fingerprint_width = self.feature_bin_count
        else:
            raise ValueError('Unknown preprocess mode "%s" (should be "mfcc",'
                            ' "average", or "micro")' % (self.preprocess))

        fingerprint_size = fingerprint_width * spectrogram_length

        self.label_count = label_count
        self.desired_samples = desired_samples
        self.window_size_samples = window_size_samples
        self.window_stride_samples = window_stride_samples
        self.spectrogram_length = spectrogram_length
        self.fingerprint_width = fingerprint_width
        self.fingerprint_size = fingerprint_size
        self.average_window_width = average_window_width
       
        
        
        
        