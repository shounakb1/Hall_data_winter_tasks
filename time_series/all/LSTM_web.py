import pickle
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.models import  Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation

from keras.layers import RepeatVector
import numpy as np
import time
from keras.layers import TimeDistributed

from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard

NAME='LSTM_web-lfr60ltd1arelu(50)-{}'.format(int(time.time()))
tensorboard=TensorBoard(log_dir='logs/{}'.format(NAME))


pickle_in=open('features(ret_seq=T).pickle','rb')
x_train,y_train,x_test,y_test=pickle.load(pickle_in)
print(y_train.shape)
model=Sequential()
model.add(LSTM((60),batch_input_shape=(None,x_train.shape[1],1),return_sequences=False))
model.add(RepeatVector(60))
model.add(LSTM((60),batch_input_shape=(None,60,1),return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.add(Activation('relu'))
# model.add(LSTM((60),batch_input_shape=(None,x_train.shape[1],1),return_sequences=False))
# model.add(Dense(60))
model.compile(loss='mean_absolute_error',optimizer='adam',metrics=['mae'])
model.summary()

print(y_train.shape)
filepath="models(lfr60ltd1arelu)/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False,save_weights_only=False,period=1)
callbacks_list = [checkpoint,tensorboard]
history=model.fit(x_train,y_train,epochs=50,validation_data=(x_test,y_test),callbacks=callbacks_list)
