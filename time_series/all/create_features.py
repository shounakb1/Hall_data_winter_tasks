import numpy as np
import pandas as pd
import pickle

df=pd.read_csv('train_1.csv')
df=df.drop(df.columns[[0]],axis=1)
df=df.sample(frac=1).reset_index(drop=True)
df=df.fillna(0)
data=np.array(df.values)
data=data.reshape(df.shape[0],df.shape[1],1)
y=data[:,-60:,:]
print(y)
x=data[:,:-60,:]

x_train=x[:int(x.shape[0]*0.9),:,:]
x_test=x[int(x.shape[0]*0.9):,:,:]
y_train=y[:int(y.shape[0]*0.9),:,:]
y_test=y[int(y.shape[0]*0.9):,:,:]

with open('features(ret_seq=T).pickle','wb') as f:
    pickle.dump([x_train,y_train,x_test,y_test],f)
