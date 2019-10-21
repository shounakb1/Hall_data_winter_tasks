import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.ensemble import RandomForestRegressor
df=pd.read_csv('googleplaystore.csv')
index=list(df['Category']).index('1.9')
df=df.drop(df.index[index])
print('index:',index)
# category=Counter(df['Category'])
df=df.reset_index(drop=True)
sum=n=i=0
for row in df['Rating']:
    if(row==row):
        sum=sum+float(row)
        n=n+1

avg=sum/n
print('avg:',avg)
i=0
for row in df['Rating']:
    if(row!=row):
        df.loc[i,'Rating']=str(avg)
    i=i+1
i=0
for row in df['Reviews']:
    try:
        float(row)
    except:
        print(df['Reviews'][i])
        df=df.drop(df.index[i])

    finally:
        i=i+1
df=df.reset_index(drop=True)
# float(df['Reviews'][index])
i=0
n=sum=0
for row in df['Size']:
    size=str(row)
    if(size[-1]=='M'):
        size=str(1000*(float(size.replace('M',''))))
    size=size.replace('k','')
    df.loc[i,'Size']=size
    try:
        float(size)

    except:
        df.loc[i,'Size']='0'
    else:
        sum=sum+float(size)
        n=n+1
    finally:
        i=i+1

avg=sum/n
print('avg:',avg)
i=0
for row in df['Size']:
    if(row=='0'):
        df.loc[i,'Size']=str(avg)
    i=i+1

i=0
for row in df['Installs']:
    installs=str(row)
    installs=installs.replace('+','')
    installs=installs.replace(',','')
    df.loc[i,'Installs']=installs
    try:
        float(installs)
        if(row!=row):
            df=df.drop(df.index[i])

    except:
        df=df.drop(df.index[i])
    finally:
        i=i+1
df=df.reset_index(drop=True)
i=0
for row in df['Price']:
    price=str(row)
    price=price.replace('$','')
    df.loc[i,'Price']=price
    try:
        float(price)
        if(row!=row):
            df=df.drop(df.index[i])

    except:
        df=df.drop(df.index[i])
    finally:
        i=i+1
df=df.reset_index(drop=True)

i=0
for row in df['Content Rating']:
    if(row!=row):
        df=df.drop(df.index[i])
    i=i+1
df=df.reset_index(drop=True)

# shuffling
df=df.sample(frac=1).reset_index(drop=True)
# content_rating=Counter(df['Content Rating'])
# print(content_rating)
df=df.dropna(subset = ['Type'])
# type=Counter(df['Type'])
# print(type)
df=df.reset_index(drop=True)
df['Day']=''
df['Month']=''
df['Year']=''
i=0
for row in df['Last Updated']:
    date=str(row)
    date=date.replace('January','01')
    date=date.replace('February','02')
    date=date.replace('March','03')
    date=date.replace('April','04')
    date=date.replace('May','05')
    date=date.replace('June','06')
    date=date.replace('July','07')
    date=date.replace('August','08')
    date=date.replace('September','09')
    date=date.replace('October','10')
    date=date.replace('November','11')
    date=date.replace('December','12')
    date=date.replace(',','')
    df['Month'][i]= date[:2]
    df['Day'][i]=date[3:-5]
    df['Year'][i]=date[-4:]
    i+=1

df=df.drop(df.columns[[0,9,13,14,15]],axis=1)
print(df)

X=df.iloc[:,[0,2,3,4,5,6,7,8,9,10]].values
Y=df.iloc[:,1].values

print(X)
print(Y)
regressor=RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(X,Y)
