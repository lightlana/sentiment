#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 21:22:15 2018

@author: lightlina
"""
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream



access_token = "1050094688408100865-121uWXPUP80LoV7nbi1bgbDueXOZb5"
access_token_secret = "LgcvVcZ1HFVAPEDb83FdlbRi6qFhqMQyESdt0881vy3Sw"
consumer_key = "OxdUkm2e77BcmhTgLXfpt4PJL"
consumer_secret = "Gk9h2zuXQpWXhkM4sVejCECdLlwWlPRDTm9Besjonj9hTtMYjp"

class StdOutListener(StreamListener):

    def on_data(self, data):
        print data
        return True

    def on_error(self, status):
        print status


if __name__ == '__main__':

    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)
    stream.filter(track=['eset', 'symantec', 'kaspersky', 'avast'])