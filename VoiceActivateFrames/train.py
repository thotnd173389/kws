#!/usr/bin/env python
# coding: utf-8

# In[1]:


from kws_streaming.layers.compat import tf
from kws_streaming.layers.modes import Modes
from kws_streaming.layers import svdf
from kws_streaming.layers import speech_features
from kws_streaming.layers.stream import Stream
from tensorflow.keras.models import model_from_json
from kws_streaming.models import utils
from utils import keyword_marvin_v1
from utils import keyword_marvin_v2

import numpy as np
import os
from datetime import datetime


# In[ ]:


X_train = np.load('x_train.npy')
Y_train = np.load('y_train.npy')
X_test = np.load('x_test.npy')
Y_test = np.load('y_test.npy')


# In[2]:


model = keyword_marvin_v2(dropout = 0.)


# In[3]:


model.summary()


# In[6]:


train_dir = './training'


# In[7]:


path_model = train_dir + '/' + model.name
if not os.path.exists(path_model):
    os.makedirs(path_model)

# serialize model to JSON
model_json = model.to_json()


with open(os.path.join(path_model, "model.json"), "w") as json_file:
    json_file.write(model_json)


# In[8]:


model.summary()

if not os.path.exists(train_dir):
    os.mkdir(train_dir)
if not os.path.exists(os.path.join(train_dir, model.name)):
    os.mkdir(os.path.join(train_dir, model.name))
    
# save model.summary()
utils.save_model_summary(model, train_dir + '/' + model.name)


# In[ ]:


save_best_weights = tf.keras.callbacks.ModelCheckpoint(
    train_dir + '/' + model.name + '/best_weights', 
    monitor='val_loss',
    mode='min', 
    verbose=0, 
    save_best_only=True,
    save_weights_only=True)


logdir = "logs/scalars/" + model.name + "/" +  datetime.now().strftime("%Y%m%d-%H%M%S")

# callbacks functions to use Tensorboard
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


# In[ ]:



loss = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001, epsilon=1e-08)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])



model.fit(X_train, Y_train, 
          validation_data = (X_test, Y_test),
          batch_size = 50, 
          verbose = 2,
          callbacks=[tensorboard_callback, save_best_weights],
          epochs = 70)

model.save_weights(train_dir + '/weights')


# In[ ]:


model.evaluate(X_test, Y_test)


# In[ ]:


model.evaluate(X_train, Y_train)


# In[ ]:

Y_test_pred = model.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(Y_test, X_test))



