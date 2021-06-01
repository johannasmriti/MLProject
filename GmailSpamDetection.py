#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install nltk


# In[4]:


import nltk


# In[5]:


nltk.download()


# In[6]:


#this program detects is an email is spam(1) or not spam(0)


# In[8]:


#importing libraries
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import string


# In[24]:


#read the csv file
df = pd.read_csv(r"C:\Users\Jegan\Downloads\SPAM-210331-134237.csv",encoding= 'unicode_escape')


# In[25]:


df.head(7)


# In[26]:


#print the shape(no of rows and columns)
df.shape


# In[27]:


#get the column names
df.columns


# In[28]:


df.drop_duplicates(inplace = True)


# In[29]:


#show shape again
df.shape


# In[31]:


df['spam'] = df['type'].map( {'spam':1, 'ham' : 0} ).astype(int)


# In[32]:


df.head(5)


# In[33]:


nltk.download('stopwords')


# In[35]:


def process_text(text):
    #1 remove punctuation
    #2 remove stopwords
    #3 return a list of clean text words
    
    #1
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    #2
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    
    #3
    return clean_words


# In[36]:


#show the tokenisation (a list of tokens also called lemmas)
df['text'].head(5).apply(process_text)


# In[40]:


#convert a collection of text to a matrix of tokens
from sklearn.feature_extraction.text import CountVectorizer
message_bow = CountVectorizer(analyzer=process_text).fit_transform(df['text'])


# In[41]:


print(message_bow)


# In[43]:


#split data into 80% training and 20% testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(message_bow, df['spam'], test_size=0.20, random_state = 0)


# In[44]:


#get shape
message_bow.shape


# In[46]:


#create and train the naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB().fit(X_train, y_train)


# In[48]:


#print the predictions
print(classifier.predict(X_train))

#print the actual values
print(y_train.values)


# In[57]:


#evaluate the model in the training data set
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
pred= classifier.predict(X_train)
print(classification_report(y_train, pred))
print()
print('Acuuracy : ',accuracy_score(y_train, pred))


# In[58]:


print(classifier.predict(X_test))
print(y_test.values)


# In[59]:


#evaluate the model in the training data set
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
pred= classifier.predict(X_test)
print(classification_report(y_test, pred))
print()
print('Acuuracy : ',accuracy_score(y_test, pred))


# In[ ]:




