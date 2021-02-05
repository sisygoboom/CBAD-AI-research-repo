# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 18:14:52 2020

@author: chris
"""

import pandas as pd
import numpy as np
from time import sleep
import tweepy
from statistics import mode
from sklearn.utils import Bunch

class StandardiseDatasets:
    def __init__(self):
        auth = tweepy.OAuthHandler(
            "eArJKT7KbozHpXbFDdHF4wLii",
            "d1abyyy0MN9NSWxrx8Acjm3cuLAbwOsko5Po35g5WZmyLRVM1G")
        auth.set_access_token(
            "418927556-i6LpwuQwSc3kytCrMkJNlDmLJ62f8JtH7mb2Kf50", 
            "jBMqq7VNfMtA1j3WHl2m9VbLXEqnyCasOg9rbhVr9tROp")
        self.twitter = tweepy.API(auth, wait_on_rate_limit=True)
        
        
    def get_all(self):
        
        #maryland = self.get_maryland()
        cornell = self.get_cornell()
        multimodal = self.get_MMHS150K()
        tweeteval = self.get_TweetEval_train()
        combined = pd.concat([maryland, cornell, multimodal, tweeteval], ignore_index=True)
        unique = combined.drop_duplicates()
        
        return unique
    
    
    # def get_maryland(self):
        
    #     data = pd.read_csv(
    #         'data/Maryland/onlineHarassmentDatasetUtf8.tdf',
    #         sep='\t',
    #         lineterminator='\r',
    #         index_col=0)
        
    #     #drop unused columns
    #     data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        
    #     map_dict = {'N': 'not hate', 'H': 'hate'}
    #     data["Code"] = data["Code"].map(map_dict)
        
    #     return data
    
    
    def get_cornell(self):
        
        data = pd.read_pickle('data/Cornell/labeled_data.p')
        
        #drop unused colummns
        del data['count']
        del data['hate_speech']
        del data['offensive_language']
        del data['neither']
        
        #data standardisation
        data.columns = ['Code', 'Tweet']
        
        map_dict= {1: 'not hate', 2: 'not hate', 0: 'hate'}
        data["Code"] = data["Code"].map(map_dict)
        
        return data
    
    
    def get_MMHS150K(self):
        
        data = pd.read_json('data/MMHS150K/MMHS150K_GT.json')
        data = data.transpose()
        del data['img_url']
        del data['labels']
        del data['tweet_url']
        
        data.columns = ['Tweet', 'Code']
        
        data['Code'] = data['Code'].map(lambda clf: mode(['not hate' if label == 'NotHate' else 'hate' for label in clf]))
        data = data.reindex(columns=['Code', 'Tweet'])
        
        return data
    
    def get_TweetEval_train(self):
        
        with open('data/TweetEval/train_text.txt', encoding='utf-8') as f:
            x = f.read().splitlines()
        with open('data/TweetEval/train_labels.txt', encoding='utf-8') as f:
            y = f.read().splitlines()
            
        y = ['not hate' if int(i) == 0 else 'hate' for i in y]
        
        d = {'Tweet': x, 'Code': y}
        
        data = pd.DataFrame(d)
        
        return data
    
    def get_TweetEval_test(self):
        
        with open('data/TweetEval/test_text.txt', encoding='utf-8') as f:
            x = f.read().splitlines()
        with open('data/TweetEval/test_labels.txt', encoding='utf-8') as f:
            y = f.read().splitlines()
        
        print(y)
        y = [False if int(i) == 0 else True for i in y]
        
        return x, y
        
    
    
    def get_tweets(self, ids):
        tweets = []
        for status_id in ids:
            try:
                tweet = self.twitter.get_status(status_id)
                print(tweet.text)
                tweets.append(tweet.text)
                    
            # rate limits exceeded
            except Exception as e:
                print(e, type(e))
                
                #status unavailable
                if e.args[0][0]['code'] in [144, 63, 179, 34]:
                    tweets.append(np.nan)
                    
                else:
                    sleep(60 * 15)
                    continue
                
        return tweets