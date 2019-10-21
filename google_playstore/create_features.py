import pickle
import numpy as np
import pandas as pd
from collections import Counter
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
# content_rating=Counter(df['Content Rating'])
# print(content_rating)

# From here creating onehot lists
category=Counter(df['Category'])
# print(category)
i=0
feat_cat=[]
for row in df['Category']:
    i=0
    feat=np.zeros(len(category))
    for cat in category:
        if(row==cat):
            feat[i]=1
        i=i+1
    feat_cat.append(feat)

i=0
content_rating=Counter(df['Content Rating'])
feat_con=[]
for row in df['Content Rating']:
    i=0
    feat=np.zeros(len(content_rating))
    for con in content_rating:
        if(row==con):
            feat[i]=1
        i=i+1
    feat_con.append(feat)
i=0
features=[]
for cat in feat_cat:
    features.append(list(cat)+list(feat_con[i]))
    i=i+1

labels=[]
for row in df['Rating']:
    labels.append(float(row))

# Normalizing
df=df.drop(df.columns[[0,1,6,8,9,10,11,12]],axis=1)
df = df.apply(pd.to_numeric)
df=(df-df.mean())/df.std()

# Normalizing

i=0
for row in df['Reviews']:
    features[i].append(row)
    i+=1
# df=df.drop(df.index[10839])
i=0
for row in df['Size']:
    features[i].append(row)
    i+=1
i=0
for row in df['Installs']:
    features[i].append(row)
    i+=1
i=0
for row in df['Price']:
    features[i].append(row)
    i+=1

# print(df.loc[[10839]])

# print(features)
i=0
for row in df['Day']:
    features[i].append(row)
    i+=1
i=0
for row in df['Month']:
    features[i].append(row)
    i+=1
i=0
for row in df['Year']:
    features[i].append(row)
    i+=1


# x_train=features[:1084*9]
# x_test=features[1084*9:10840]
# y_train=labels[:1084*9]
# y_test=labels[1084*9:10840]

print(len(labels))
print(len(features))


with open('features_with_date.pickle','wb') as f:
    pickle.dump([features,labels],f)
