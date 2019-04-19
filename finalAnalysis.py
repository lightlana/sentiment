#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 19:28:41 2018

@author: lightlina
"""
import json
import collections
import re
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter

#load the testing data and count the number of tweets

tweets_data_path = '/home/lightlina/Desktop/MEGAsync/Sentiment analysis/Test_set/skoricafinal.txt'

tweets_data = []
tweets_file = open(tweets_data_path, "r")


for line in tweets_file:
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
    except:
        continue

print(len(tweets_data))


"""
load into dataframe
first create an empty dataframe, then load data from tweets_data into it
"""

tweets = pd.DataFrame()

tweets['text'] = list(map(lambda tweet: tweet['text'], tweets_data))
tweets['lang'] = list(map(lambda tweet: tweet['lang'], tweets_data))
tweets['country'] = list(map(lambda tweet: tweet['place']['country'] if tweet['place'] != None else None, tweets_data))


#graph for the most common countries

tweets_by_country = tweets['country'].value_counts()

fig, ax = plt.subplots()
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=10)
ax.set_xlabel('Countries', fontsize=15)
ax.set_ylabel('Number of tweets', fontsize=15)
ax.set_title('Top 5 countries', fontsize=15, fontweight='bold')
tweets_by_country[:5].plot(ax=ax, kind='bar', color='blue')


#choose only tweets in english and tranform text to lower case
tweets['text'] = tweets['text'].str.lower()
tweets = tweets.loc[tweets['lang'] == 'en']


"""
create new columns that indicate if the name of company is in the text
print the number of tweets with the name of each company
"""
def word_in_text(word, text):
    match = re.search(word, text)
    if match:
        return True
    return False


tweets['eset'] = tweets['text'].apply(lambda tweet: word_in_text('eset', tweet))
tweets['avast'] = tweets['text'].apply(lambda tweet: word_in_text('avast', tweet))
tweets['kaspersky'] = tweets['text'].apply(lambda tweet: word_in_text('kaspersky', tweet))
tweets['symantec'] = tweets['text'].apply(lambda tweet: word_in_text('symantec', tweet))

total_sample = len(tweets)
number_eset = tweets['eset'].value_counts()[True]
number_avast = tweets['avast'].value_counts()[True]
number_kasp = tweets['kaspersky'].value_counts()[True]
number_sym = tweets['symantec'].value_counts()[True]

print("Number of ESET tweets:", number_eset)
print("Number of AVAST tweets:", number_avast)
print("Number of Kaspersky tweets:", number_kasp)
print("Number of Symantec tweets:", number_sym)


"""
Graph: Compare the number of tweets for companies

Setting axis labels and ticks
"""
scr_comp = ['eset', 'avast', 'kaspersky', 'symantec']
tweets_by_scr_comp = [number_eset, number_avast, number_kasp, number_sym]

x_pos = list(range(len(scr_comp)))
width = 0.8
fig, ax = plt.subplots()
plt.bar(x_pos, tweets_by_scr_comp, width, alpha=1, color='g')

ax.set_ylabel('Number of tweets', fontsize=15)
ax.set_title('Ranking: Eset vs. Avast vs. Kaspersky vs. Symantec', fontsize=10, fontweight='bold')
ax.set_xticks([p + 0.4 * width for p in x_pos])
ax.set_xticklabels(scr_comp)
plt.grid()





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
    phrase = re.sub('u[\U0001F60A-\U0001F604-\U0001F603]', "smilley", phrase)
    #wink
    phrase = re.sub(u'[\U0001F609]', 'winkie', phrase)
    #sad
    phrase = re.sub(u'[\U0001F614-\U0001F61E-\U0001F622-\U0001F62D]', "saddie", phrase)
    return phrase



#fill NAs
tweets['text'] = tweets['text'].fillna(' ')
#apply function
tweets['text'] = tweets['text'].apply(lambda x: decontracted(x))
#
tweets['text'] = tweets['text'].apply(lambda x: emoticons(x))
# remove links
tweets['text'] = tweets['text'].replace(r'http\S+', ' ', regex=True).replace(r'www\S+', ' ', regex=True)
#remove user names
tweets['text'] = tweets['text'].replace("@[\w]*", ' ', regex=True)
#remove hashtags
tweets['text'] = tweets['text'].replace('#[\w+]', ' ', regex=True)
# characters to lower
tweets['text'] = tweets['text'].apply(lambda x: x.lower()).apply(lambda x: x.replace('#', ' '))
#only normal characters
tweets['text'] = tweets['text'].str.replace("[^a-zA-Z#]", " ")
#remove whitespaces
tweets['text'] = tweets['text'].apply(lambda x: x.strip())




#Tokenize, delete stop words and join




stop_words = set(stopwords.words('english'))
tweets['text'] = tweets['text'].str.split()
tweets['text'] = tweets['text'].apply(lambda x: [item for item in x if item not in stop_words])
tweets['text'] = tweets['text'].apply(lambda x: ' '.join(x))





"""
Number of 10 most common words in each company
Create a function Count
Print a table with most common words by corpus and companies
"""

def Count(text):
    count = Counter(word for line in text
                    for word in line.split())
    return(count.most_common(20))


eset = tweets.loc[tweets['eset'] == True]
kaspersky = tweets.loc[tweets['kaspersky'] == True]
avast = tweets.loc[tweets['avast'] == True]
symantec = tweets.loc[tweets['symantec'] == True]


t = PrettyTable()
t.add_column("Corpus", Count(tweets['text']))
t.add_column("ESET", Count(eset['text']))
t.add_column('Kaspersky', Count(kaspersky['text']))
t.add_column('Avast', Count(avast['text']))
t.add_column('Symantex', Count(symantec['text']))
print(t)


#count the number of tweets with emoticons
tweets['smilley'] = tweets['text'].apply(lambda tweet: word_in_text('smilley', tweet))
tweets['sad'] = tweets['text'].apply(lambda tweet: word_in_text('saddie', tweet))

tweets['smilley'].value_counts()[True]
tweets['sad'].value_count()[True]



"""
load model with pickle
transform test data with tokenizer trained on training data

"""
filename = 'LSTM_model.sav'
infile = open(filename, 'rb')
model = pickle.load(infile)
infile.close()


data = pd.read_csv('clasifier_final.csv', sep=';', encoding='latin1')
data.tidy_content = data.tidy_content.astype(str)
train = data['tidy_content']


sentences = tweets['text']
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(train.values)
X = tokenizer.texts_to_sequences(sentences.values)
X = pad_sequences(X, maxlen=25)


#apply trained LSTM model on the testing data
#predict emotion labels
emocie = model.predict(X)
emocieClass = model.predict_classes(X)


def CountFrequency(arr):
    return collections.Counter(arr)

print(CountFrequency(emocieClass))

tweets['emotions'] = emocieClass


#export dataframe to csv
tweets.to_csv("companies_final.csv", sep=';', encoding='utf-8')
