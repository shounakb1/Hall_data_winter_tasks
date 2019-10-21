import numpy as np
import pandas as pd
import pickle
from imblearn.over_sampling import SMOTE
from collections import Counter

df=pd.read_csv('creditcard.csv')

df=df.sample(frac=1).reset_index(drop=True)


y=np.array(df['Class'])

df=df.drop(df.columns[[30]],axis=1)
df = df.apply(pd.to_numeric)
df=(df-df.mean())/df.std()
x=np.array(df.values)
print(Counter(y))

#
#
x=x[:int(x.shape[0]*0.7),:]
x_test=x[int(x.shape[0]*0.7):,:]
y=y[:int(y.shape[0]*0.7)]
y_test=y[int(y.shape[0]*0.7):]

sm = SMOTE()
x_train, y_train = sm.fit_resample(x, y)
print(Counter(y_test))
#
with open('features.pickle','wb') as f:
    pickle.dump([x_train,y_train,x_test,y_test],f)
