#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

model=tf.keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])

for layer in model.layers:
    weights = layer.get_weights()
    weights = np.array([[0.6519221]], dtype=np.float32)

model.compile(optimizer='sgd', loss='mse')
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=np.float32)

plt.scatter(xs, ys)

model.fit(xs, ys, epochs=500)
print(model.predict([10.0]))


# In[ ]:




