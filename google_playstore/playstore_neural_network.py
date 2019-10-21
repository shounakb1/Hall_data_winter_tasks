from __future__ import absolute_import, division, print_function
import tensorflow as tf
import pickle
import pathlib
import numpy as np
from tensorflow import keras
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import time
from keras import regularizers

NAME='playstore_neural_network(l2reg)(with_date)-128*3(50)-{}'.format(int(time.time()))
tensorboard=TensorBoard(log_dir='logs/{}'.format(NAME))



def custom_activation(x):
    return (K.sigmoid(x) * 5)

get_custom_objects().update({'custom_activation': Activation(custom_activation)})

pickle_in=open('features_with_date.pickle','rb')
train_x,train_y=pickle.load(pickle_in)
model=tf.keras.models.Sequential()
# model.add(Activation(custom_activation))
tx=[]

for arr in train_x:
    tx.append(np.array(arr))
ty=np.array(train_y)
tx=np.array(tx)



model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,kernel_regularizer=regularizers.l2(0.01),activation=tf.nn.relu))


model.add(tf.keras.layers.Dense(1,activation=custom_activation))
# model.add(tf.keras.layers.Dense(1))
optimiser=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='mse',optimizer=optimiser,metrics=['mae','mse'])
history=model.fit(tx,ty,epochs=50,validation_split=0.3,verbose=0,callbacks=[tensorboard])
hist=pd.DataFrame(history.history)
hist['epoch']=history.epoch
print(hist)
