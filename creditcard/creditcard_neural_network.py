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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

NAME='creditcard_neural_network-256*2(50)-{}'.format(int(time.time()))
tensorboard=TensorBoard(log_dir='logs/{}'.format(NAME))

def auroc(y_true, y_pred):
    try:
        return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)
    except ValueError:
        pass

pickle_in=open('features.pickle','rb')
x_train,y_train,x_test,y_test=pickle.load(pickle_in)
model=tf.keras.models.Sequential()
# model.add(Activation(custom_activation))
# tx=[]
#
# for arr in train_x:0
#     tx.append(np.array(arr))
# ty=np.array(train_y)
# tx=np.array(tx)
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)


model.add(tf.keras.layers.Dense(256,activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(128,kernel_regularizer=regularizers.l2(0.01),activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(256,activation=tf.nn.relu))


# model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))


model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy', auroc])
history=model.fit(x_train,y_train,epochs=50,batch_size=398050,validation_data=(x_test,y_test),verbose=1,callbacks=[tensorboard])
hist=pd.DataFrame(history.history)
hist['epoch']=history.epoch
print(hist)
