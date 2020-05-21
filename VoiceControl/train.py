#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 20:38:09 2020

@author: thorius
"""
# %%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import tensorflow.compat.v1 as tf
import input_data
import logging
from kws_streaming.models import utils
from DataSettings import DataSettings
from TrainingSettings import TrainingSettings
from utils import E2E_1stage_v1
from utils import E2E_1stage_v2
from utils import E2E_1stage_v3
from utils import E2E_1stage_v4
from utils import E2E_1stage_v5
from utils import E2E_1stage_v6
from utils import E2E_1stage_v7
from utils import E2E_1stage_v8
from utils import E2E_1stage_v9
from utils import E2E_1stage_v8_vl_0_4
from utils import E2E_1stage_v8_vl_0_3

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
    window_size_ms = 40.0,
    window_stride_ms = 20.0,
    dct_num_features = 40,
    mel_num_bins = 80,
    mel_upper_edge_hertz = 7000,
    silence_percentage = 6.0,
    unknown_percentage = 6.0,
    background_volume = 0.4,
    wanted_words = 'on,off,up,down,zero,one,two,three,four,five,six,seven,eight,nine')


training_settings = TrainingSettings()


# %%
time_shift_samples = int((data_settings.time_shift_ms * data_settings.sample_rate) / 1000)


# %%
audio_processor = input_data.AudioProcessor(data_settings)

# %%
model = E2E_1stage_v8_vl_0_4(input_shape=(16000), data_settings=data_settings)
logging.info(model.summary())

# save model summary
utils.save_model_summary(model, training_settings.train_dir)




loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(epsilon=training_settings.optimizer_epsilon)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
# %%

train_writer = tf.summary.FileWriter(training_settings.summaries_dir + model.name + '/train', sess.graph)
validation_writer = tf.summary.FileWriter(training_settings.summaries_dir  + model.name + '/validation')

sess.run(tf.global_variables_initializer())

start_step = 1

logging.info('Training from step: %d ', start_step)

# Save graph.pbtxt.
tf.train.write_graph(sess.graph_def, training_settings.train_dir + model.name, 'graph.pbtxt')

# Save list of words.
with tf.io.gfile.GFile(os.path.join(training_settings.train_dir + model.name, 'labels.txt'), 'w') as f:
  f.write('\n'.join(audio_processor.words_list))

# Training loop.
best_accuracy = 0.0
training_steps_max = 60000
learning_rate_value = 0.0005
for training_step in range(start_step, training_steps_max + 1):

  # Pull the audio samples we'll use for training.
  train_fingerprints, train_ground_truth = audio_processor.get_data(
      data_settings.batch_size, 0, data_settings, data_settings.background_frequency,
      data_settings.background_volume, time_shift_samples, 'training', data_settings.resample,
      sess)

  tf.keras.backend.set_value(model.optimizer.lr, learning_rate_value)
  result = model.train_on_batch(train_fingerprints, train_ground_truth)

  summary = tf.Summary(value=[
      tf.Summary.Value(tag='accuracy', simple_value=result[1]),
  ])
  
  train_writer.add_summary(summary, training_step)


  logging.info(
      'Step #%d: rate %f, accuracy %.2f%%, cross entropy %f',
      *(training_step, learning_rate_value, result[1] * 100, result[0]))

  is_last_step = (training_step == training_steps_max)
  if (training_step % training_settings.eval_step_interval) == 0 or is_last_step:
    set_size = audio_processor.set_size('validation')
    set_size = int(set_size / data_settings.batch_size) * data_settings.batch_size
    total_accuracy = 0.0
    count = 0.0
    for i in range(0, set_size, data_settings.batch_size):
      validation_fingerprints, validation_ground_truth = (
          audio_processor.get_data(data_settings.batch_size, i, data_settings, 0.0, 0.0, 0,
                                   'validation', 0.0, sess))

      # Run a validation step and capture training summaries for TensorBoard
      # with the `merged` op.
      result = model.test_on_batch(validation_fingerprints,
                                   validation_ground_truth)

      summary = tf.Summary(value=[
          tf.Summary.Value(tag='accuracy', simple_value=result[1]),])

      validation_writer.add_summary(summary, training_step)

      total_accuracy += result[1]
      count = count + 1.0

    total_accuracy = total_accuracy / count
    logging.info('Step %d: Validation accuracy = %.2f%% (N=%d)',
                 *(training_step, total_accuracy * 100, set_size))

    model.save_weights(training_settings.train_dir + 'train/' + model.name + '/' +  
                       str(int(best_accuracy * 10000)) + 'weights')

    # Save the model checkpoint when validation accuracy improves
    if total_accuracy > best_accuracy:
      best_accuracy = total_accuracy
      # overwrite the best model weights
      model.save_weights(training_settings.train_dir + model.name + '/best_weights')
    logging.info('So far the best validation accuracy is %.2f%%',
                 (best_accuracy * 100))

