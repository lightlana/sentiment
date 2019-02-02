#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 19:28:41 2018

@author: lightlina
"""

import json
import pandas as pd
import matplotlib.pyplot as plt

tweets_data_path = '/home/lightlina/Desktop/skorica (3).txt'


tweets_data = []
tweets_file = open(tweets_data_path, "r")
for line in tweets_file:
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
    except:
        continue

#kolko tweetov     
print len(tweets_data)

#nacitanie do dataframu
#najprv sa vytvori prazdny dataframe, 
tweets = pd.DataFrame()

tweets['text'] = map(lambda tweet: tweet['text'], tweets_data)
tweets['lang'] = map(lambda tweet: tweet['lang'], tweets_data)
tweets['country'] = map(lambda tweet: tweet['place']['country'] if tweet['place'] != None else None, tweets_data)

#graf na najcastejsie krajiny, ktore tweetovali
tweets_by_country = tweets['country'].value_counts()

fig, ax = plt.subplots()
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=10)
ax.set_xlabel('Countries', fontsize=15)
ax.set_ylabel('Number of tweets' , fontsize=15)
ax.set_title('Top 5 countries', fontsize=15, fontweight='bold')
tweets_by_country[:5].plot(ax=ax, kind='bar', color='blue')


#oznacia sa tie tweets, ktore obsahuje urcite slova
#prida sa stlpec, ktory znaci, ci sa dane slovo v tweete nachadza

import re

def word_in_text(word, text):
    word = word.lower()
    text = text.lower()
    match = re.search(word, text)
    if match:
        return True
    return False

#vyber tweets s klucovymi slovami
tweets['google'] = tweets['text'].apply(lambda tweet: word_in_text('google', tweet))
tweets['ibm'] = tweets['text'].apply(lambda tweet: word_in_text('ibm', tweet))
tweets['dell'] = tweets['text'].apply(lambda tweet: word_in_text('dell', tweet))

#kolko tweetov je v ramci kolkych firiem
print tweets['google'].value_counts()[True]
print tweets['ibm'].value_counts()[True]
print tweets['dell'].value_counts()[True]


#porovnanie tweets pre rozne kategorie
prg_langs = ['python', 'javascript', 'ruby']
tweets_by_prg_lang = [tweets['python'].value_counts()[True], tweets['javascript'].value_counts()[True], tweets['ruby'].value_counts()[True]]

x_pos = list(range(len(prg_langs)))
width = 0.8
fig, ax = plt.subplots()
plt.bar(x_pos, tweets_by_prg_lang, width, alpha=1, color='g')

# Setting axis labels and ticks
ax.set_ylabel('Number of tweets', fontsize=15)
ax.set_title('Ranking: python vs. javascript vs. ruby (Raw data)', fontsize=10, fontweight='bold')
ax.set_xticks([p + 0.4 * width for p in x_pos])
ax.set_xticklabels(prg_langs)
plt.grid()




tweets.to_csv("skuska_tweets", sep='\t', encoding='utf-8')
