#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# In[2]:


from PIL import Image
img=Image.open('irisimg.jpg')
print(img)


# In[3]:


img


# In[4]:


df=pd.read_csv('iris.csv')
df


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.shape


# In[8]:


df.value_counts()


# In[9]:


df.isnull()


# In[10]:


df.isnull().sum()


# In[11]:


df.hist()
plt.show()


# In[12]:


c=['pink','yellow','black']
f=['Iris-versicolor','Iris-virginica','Iris-setosa']
for i in range(3):
    x=df[df['Species']==f[i]]
    plt.scatter(x['SepalLengthCm'],x['SepalWidthCm'],c=c[i],label=f[i])
plt.xlabel('Sepallength(cm)')
plt.ylabel('Sepalwidth(cm)')
plt.legend()


# In[13]:


c=['pink','yellow','black']
f=['Iris-versicolor','Iris-virginica','Iris-setosa']
for i in range(3):
    x=df[df['Species']==f[i]]
    plt.scatter(x['PetalLengthCm'],x['PetalWidthCm'],c=c[i],label=f[i])
plt.xlabel('Petallength(cm)')
plt.ylabel('Petalwidth(cm)')
plt.legend()


# In[14]:


c=['pink','yellow','black']
f=['Iris-versicolor','Iris-virginica','Iris-setosa']
for i in range(3):
    x=df[df['Species']==f[i]]
    plt.scatter(x['SepalLengthCm'],x['PetalWidthCm'],c=c[i],label=f[i])
plt.xlabel('Sepallength(cm)')
plt.ylabel('Petalwidth(cm)')
plt.legend()


# In[15]:


c=['pink','yellow','black']
f=['Iris-versicolor','Iris-virginica','Iris-setosa']
for i in range(3):
    x=df[df['Species']==f[i]]
    plt.scatter(x['PetalLengthCm'],x['SepalWidthCm'],c=c[i],label=f[i])
plt.xlabel('Petallength(cm)')
plt.ylabel('Sepalwidth(cm)')
plt.legend()


# In[16]:


df.corr()


# In[17]:


sns.heatmap(df.corr())


# In[18]:


from sklearn.preprocessing import LabelEncoder


# In[19]:


e=LabelEncoder()
df['Species']=e.fit_transform(df['Species'])


# In[20]:


df.head(10)


# In[21]:


df.tail()


# In[22]:


x=df.drop(columns=['Species'])
y=df['Species']


# In[23]:


x


# In[24]:


y


# In[25]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)


# In[26]:


lg=LogisticRegression()


# In[27]:


lg.fit(x_train,y_train)


# In[28]:


out=lg.score(x_test,y_test)*100
out


# In[29]:


print("accuracy of the logistic regression is" , out)


# In[30]:


y_predict=lg.predict(x_test)
print(y_predict)


# In[31]:


d=DecisionTreeClassifier()
d.fit(x_train,y_train)


# In[32]:


outscore=d.score(x_test,y_test)*100
print(outscore)


# In[33]:


print('Accuracy of Decision Tree is',outscore)


# In[34]:


from sklearn.metrics import confusion_matrix


# In[35]:


c=confusion_matrix(y_test,y_predict)
print(c)


# In[ ]:





# In[ ]:




