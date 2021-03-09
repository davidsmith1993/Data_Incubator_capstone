# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:35:56 2020

@author: Pablo
"""
# Importing necessary library
import pandas as pd
import numpy as np
import nltk
import os
import nltk.corpus
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re

#nltk.download('punkt')



dfall = pd.DataFrame()


#Import Datasets from the pulled comments. This takes a while to run
dfs = ['C:/Users\Pablo\Documents\Data Challenges\Reddit\Datasets/Raw/otherandnews_comments_DI.csv',
      'C:/Users\Pablo\Documents\Data Challenges\Reddit\Datasets/Raw/otherandnews_comments_DI2.csv',
      'C:/Users\Pablo\Documents\Data Challenges\Reddit\Datasets/Raw/religion_comments_DI.csv',
      'C:/Users\Pablo\Documents\Data Challenges\Reddit\Datasets/Raw/politics_comments_DI.csv']



for x in dfs:
    df = pd.read_csv(x)
    #Start with a smaller sample
    #df = df.sample(50000, replace=True)
    df['body'].dropna(inplace=True)
    
    """
    #Tokenize
    def custom_tokenize(text):
        if not text:
            print('The text to be tokenized is a None type. Defaulting to blank string.')
            text = ''
        return word_tokenize(text)
    df['token_text'] = df.body.apply(custom_tokenize)
    """
    
    
    
    
    my_stopwords = nltk.corpus.stopwords.words('english')
    word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem
    my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'
    
    # cleaning master function
    def clean_tweet(tweet, bigrams=False):
        tweet = tweet.lower() # lower case
        tweet = re.sub('['+my_punctuation + ']+', ' ', tweet) # strip punctuation
        tweet = re.sub('\s+', ' ', tweet) #remove double spacing
        tweet = re.sub('([0-9]+)', '', tweet) # remove numbers
        tweet_token_list = [word for word in tweet.split(' ')
                                if word not in my_stopwords] # remove stopwords
    
        #tweet_token_list = [word_rooter(word) if '#' not in word else word
                            #for word in tweet_token_list] # apply word rooter
        if bigrams:
            tweet_token_list = tweet_token_list+[tweet_token_list[i]+'_'+tweet_token_list[i+1]
                                                for i in range(len(tweet_token_list)-1)]
        tweet = ' '.join(tweet_token_list)
        return tweet
    
    
    df['clean'] = df.body.apply(clean_tweet)
    df['clean'].dropna(inplace=True)
    

    
    
    
    """ Sentiment Analyses"""
    

    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk.download('vader_lexicon')
    nltk.download('punkt')
    #Our goal here is to conduct sentiment analyses of the comments
    sid = SentimentIntensityAnalyzer()
    df2 = pd.DataFrame()
    
    
    
    
    
    for i in df['clean']:
        scores = sid.polarity_scores(i)
    
        #for key in sorted(scores):
            #print('{0}: {1} '.format(key, scores[key]), end='')
    
        if scores["compound"] >= 0.05:
            df2.loc[i,'sentiment1'] = 'positive'
    
        elif scores["compound"] <= -0.05:
            df2.loc[i,'sentiment1'] = 'negative'
        else:
            df2.loc[i,'sentiment1'] = 'neutral'
     
    df.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)      
    df = pd.concat([df, df2], axis=1)
    dfall = dfall.append(df, ignore_index=True)

#save everything to one csv    
dfall.to_csv(r'C:/Users/Pablo/Documents/Data Challenges/Reddit/Datasets/NLP/comments_NLP_sent.csv') 