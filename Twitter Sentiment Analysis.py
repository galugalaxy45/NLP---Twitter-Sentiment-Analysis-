#!/usr/bin/env python
# coding: utf-8

# # <span style = "color:green"> Twitter Sentiment Analysis </span>

# ***

# Sentiment analysis refers to identifying as well as classifying the sentiments that are expressed in the text source. Tweets are often useful in generating a vast amount of sentiment data upon analysis. These data are useful in understanding the opinion of the people about a variety of topics.
# 
# Therefore we need to develop an Automated Machine Learning Sentiment analysis Model in order to compute the customer perception. Due to the presence of non-useful characters (collectively termed as the noise) along with useful data, it becomes difficult to implement models on them.
# 
# Here, We aim to analyze the sentiment of the tweets provided in the dataset by developing a machine learning pipeline involving the use of SVM classifier along with using Term Frequency-Inverse Document Frequency(TF-IDF). 
# 
# The dataset consist of 13870 tweets that have been extracted using the Twitter API. The dataset contains various columns but for this specific problem, we would only be using
#    * Sentiment - Positive, Negative, Neutral
#    * Text - Tweet

# ## Let's get Started

# ### Import Necessay Libraries

# In[59]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk


# ### Read the dataset

# In[104]:


data=pd.read_csv('twitter.csv',encoding='ISO-8859-1')
data


# ### View head

# In[105]:


data.head()


# ### View info of the dataset

# In[106]:


data.info()


# ### Drop all columns exept 'text' and 'sentiment'

# In[107]:


data=data[['text','sentiment']]


# In[108]:


data.head()


# ### Check all the unique values in Sentiment

# In[109]:


data.columns


# In[110]:


data['sentiment'].unique()


# ### Convert Neutral to 0, Positive to 1 and Negative to -1

# In[111]:


def convert(x):
    if x=='Neutral':
        return 0
    elif x=='Positive':
        return 1
    else:
        return -1


# In[112]:


data['sentiment']=data['sentiment'].apply(convert)


# In[113]:


data.head()


# ### Check for missing values

# In[114]:


data.isna().sum()


# ### Check for Duplicates

# In[115]:


data.duplicated().sum()


# ### Drop duplicate rows

# In[116]:


data.drop_duplicates(keep='first',inplace=True)


# In[117]:


data.duplicated().sum()


# ### View some of the tweets

# In[118]:


for Sentence in data['text'].head(10):
    print(Sentence)


# ### Exploratory Data Analysis

# ### Plot a countplot of sentiment

# In[119]:


sns.countplot(x='sentiment',data=data)
plt.show()


# In[120]:


data['sentiment'].unique()


# ### Plot a piechart to show the percentile representation of sentiments

# In[121]:


plt.pie(data['sentiment'].value_counts(), labels = ['Negative', 'Neutral','Positive'], autopct = '%0.2f')
plt.show()


# ### Define a function that preprocess the tweets

# ie, 
# * Remove all special characters
# * Remove any stopwords
# * Lemmatize the words

# In[122]:


from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# In[123]:


def preprocess(sentence):
    
    #removes all the special characters and split the sentence at spaces
    text=re.sub(r'[^0-9a-zA-Z]+',' ',sentence).split()
    
    # converts words to lowercase and removes any stopwords
    words = [x.lower() for x in text if x not in stopwords.words('english')]
    
    # Lemmatize the words
    lemma = WordNetLemmatizer()
    word = [lemma.lemmatize(word,'v') for word in words ]
    
    # convert the list of words back into a sentence
    word = ' '.join(word)
    return word
        


# In[124]:


preprocess(data.text[0])


# ### Apply the function to our tweets column

# In[125]:


data['text'] = data['text'].apply(preprocess)


# ### Print some of the tweets after preprocessing

# In[127]:


for i in range(10):
    print(data.iloc[i]['text'])
    print()


# ### Assign X and y variables

# In[129]:


X=data['text']
y=data['sentiment']


# ### Transform X variable(tweets) using TF-IDF Vectorizer

# In[130]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[132]:


vectorizer=TfidfVectorizer(ngram_range=(2,2))


# In[133]:


X=vectorizer.fit_transform(X)


# In[134]:


type(X)


# ### Split the data into training and testing set

# In[135]:


from sklearn.model_selection import train_test_split


# In[136]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# ### Check the shape of X_train and X_test

# In[137]:


X_train.shape


# In[138]:


X_test.shape


# ### Create a SVM Model

# In[139]:


from sklearn.svm import SVC


# In[140]:


model=SVC()


# ### Train the model

# In[141]:


model.fit(X_train,y_train)


# ### Check the score of the training set

# In[142]:


model.score(X_train,y_train)


# ### Make prediction with X_test

# In[147]:


y_pred=model.predict(X_test)


# ### Check the accuracy of our prediction

# In[146]:


from sklearn import metrics


# In[148]:


metrics.accuracy_score(y_test,y_pred)


# ### Plot confusion matrix on heatmap

# In[150]:


sns.heatmap(metrics.confusion_matrix(y_test,y_pred),annot=True,fmt='d')
plt.show()


# ### Print Classification report

# In[152]:


print(metrics.classification_report(y_test,y_pred))


# ***
