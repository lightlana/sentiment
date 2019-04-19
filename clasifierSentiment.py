#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 13:45:28 2019

@author: lightlina
"""
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 20:45:58 2018

@author: lightlina
"""

import pandas as pd
import re
import copy
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

#load datasets for fear and anger
anger_fear = pd.read_csv('fear_anger', sep =';')

#load dataset, find columns names and unique emotions labels
#rename tweet_id to ID
data1  = pd.read_csv('text_emotion.csv')
#list(data1.columns.values)
#data1['sentiment']
data1.sentiment.unique()
data1.rename(columns={'tweet_id':'ID'}, inplace=True)


#load dataset, find columns names and unique emotions lables
#rename sentence to content, emotion to sentiment and id to ID
data2 = pd.read_csv('primary-plutchik-wheel-DFE.csv') 
#list(data2.columns.values)
#data2['emotion']
data2.emotion.unique()
data2.rename(columns={'sentence':'content', 
                      'emotion':'sentiment', 
                      'id':'ID'}, inplace=True)

#choose rows from data 1 and data 2 that are the same
data1_emotions= data1.loc[data1['sentiment'].isin(['sadness', 
                               "neutral", 
                               "suprise",
                               'love', 
                               'happiness', 
                               'anger', 
                               'enthusiasm'])]



data2_emotions=  data2.loc[data2['sentiment'].isin(['Sadness', 
                           "Neutral", 
                           'Suprise', 
                           'Love', 
                           'Joy', 
                           'Aggression', 
                           'Fear', 'Optimism'])]


#all datasets into one
frames = [data1_emotions, 
          data2_emotions, 
          anger_fear]

result = pd.concat(frames)
result.head()
list(result.columns.values)

result['sentiment'] = result['sentiment'].str.lower()
result.sentiment.replace(['joy', 'aggression'], ['happiness', 'anger'], inplace=True)

print(result.groupby('sentiment').size())


clasifier = result.loc[result['sentiment'].isin([
                        'anger', 'fear', 
                       'happiness', 'love',
                       'neutral', 'sadness'])]



print(clasifier.groupby('sentiment').size())


"""
Data correction
Fill NAs
replace links and user names, only alphabets, to lower
hashtags remove
"""

porter = PorterStemmer()

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def emoticons(phrase):
    #happy
    phrase = re.sub('u[\U0001F60A-\U0001F604-\U0001F603]', "smille", phrase) 
    #wink
    phrase = re.sub(u'[\U0001F609]','wink', phrase)    
    #sad
    phrase = re.sub(u'[\U0001F614-\U0001F61E-\U0001F622-\U0001F62D]', "sad", phrase)
    return phrase
    


clasifier['tidy_content'] = copy.deepcopy(clasifier['content'])
#fill NAs
clasifier['tidy_content'] = clasifier['tidy_content'].fillna(' ')
#apply function

clasifier['tidy_content'] = clasifier['tidy_content'].apply(lambda x: decontracted(x))
#clasifier['tidy_content'] = clasifier['tidy_content'].apply(lambda x: emoticons(x))
# remove links
clasifier['tidy_content'] = clasifier['tidy_content'].replace(r'http\S+', ' ', regex=True).replace(r'www\S+', ' ', regex=True)
#remove user names
clasifier['tidy_content'] = clasifier['tidy_content'].replace("@[\w]*", ' ', regex=True)
#remove hashtags
clasifier['tidy_content'] = clasifier['tidy_content'].replace('#[\w+]',' ', regex = True)
# characters to lower
clasifier['tidy_content'] = clasifier['tidy_content'].apply(lambda x: x.lower()).apply(lambda x: x.replace('#',' '))
#only normal characters
clasifier['tidy_content'] = clasifier['tidy_content'].str.replace("[^a-zA-Z#]", " ")
#remove whitespaces
clasifier['tidy_content'] = clasifier['tidy_content'].apply(lambda x: x.strip())
#tokenize and delete stop words
clasifier['tidy_content'] = clasifier['tidy_content'].str.split()
stop_words = set(stopwords.words('english'))
clasifier['tidy_content'] = clasifier['tidy_content'].apply(lambda x: [item for item in x if item not in stop_words])
# join tokens
clasifier['tidy_content'] =clasifier['tidy_content'].apply(lambda x: ' '.join(x))
#stemming
clasifier['tidy_content'] = clasifier['tidy_content'].apply(lambda x: porter.stem(x))

# choose only relevant columns
clasifier = clasifier[['tidy_content', 'sentiment']]

#add column where sentiment is represennted as nubber
clasifier.sentiment = pd.Categorical(clasifier.sentiment)
clasifier['labels'] = clasifier.sentiment.cat.codes


def replace(tweet):
    tweet =  tweet.fillna(' ')
    tweet = tweet.replace(r'http\S+', ' ', regex=True).replace(r'www\S+', ' ', regex=True)
    tweet = tweet.replace("@[\w]*", ' ', regex=True)
    tweet = tweet.str.replace("[^a-zA-Z#]", " ")
    


#save as csv 
clasifier.to_csv("clasifier_final.csv", sep=';', encoding='utf-8')




