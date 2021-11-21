#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow import keras


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[ ]:


(X_train,y_train),(X_test,y_test)=keras.datasets.mnist.load_data()


# In[ ]:


len(X_test)


# In[ ]:


X_train[0].shape


# In[ ]:


plt.matshow(X_train[9])


# In[ ]:


X_train=X_train/255
X_test=X_test/255


# In[ ]:


X_train


# In[ ]:


X_train_flattened=X_train.reshape(len(X_train),28*28)
X_test_flattened =X_test.reshape(len(X_test),28*28)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


model.evaluate(X_test_flattened , y_test)


# In[ ]:


y_pred=model.predict(X_test_flattened)


# In[ ]:


y_pred[1]


# In[ ]:


np.argmax(y_pred[1])


# In[ ]:


y_predicted=[np.argmax(i) for i in y_pred]


# In[ ]:


y_test[:5]


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_predicted)


# In[ ]:


import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner.tuners import RandomSearch


# In[ ]:


def build(hp):
  m = keras.Sequential()
  for i in range(hp.Int('num_layers',2,20)):
      m.add(keras.layers.Dense(
          hp.Int('units' + str(i),
                         min_value=8,
                        max_value=500),
          activation='relu'))
  m.add(keras.layers.Dense(1, activation='sigmoid'))
  m.compile(
      optimizer=keras.optimizers.Adam(
      hp.Choice('learning_rate',[1e-2,1e-3,1e-4])) ,     
      loss='mean_absolute_error' ,
  metrics=['mean_absolute_error'])
  return model


# In[ ]:


t = kt.RandomSearch(
    build,
    objective='val_mean_absolute_error',
    max_trials=10)


# In[ ]:


t.search_space_summary()


# In[ ]:


t.search(X_train_flattened, y_train, epochs=10, validation_data=(X_test_flattened, y_test))
t.results_summary()


# In[ ]:


model=keras.Sequential([
    keras.layers.Dense(100,input_shape=(784,),activation='relu'),
    keras.layers.Dense(10,activation='relu'),
    keras.layers.Dense(10,activation='sigmoid')
    
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(X_train_flattened,y_train,epochs=10)


# In[ ]:




