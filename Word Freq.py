# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 13:09:11 2020

@author: Pablo
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import collections

import nltk
from nltk.corpus import stopwords
import re
import networkx

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('C:/Users\Pablo\Documents\Data Incubator/Reddit/New/Data/df_all_cleaned.csv')
#df = df.sample(500000)

df['body'] = df['body'].astype(str)

#Clean the comment text data

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
    tweet = tweet_token_list
    return tweet


#Most common words overall
df['clean_split'] = df.body.apply(clean_tweet)
all_words= list(itertools.chain(*df['clean_split']))
x = ([i for i in all_words if i  != ''])
x = ([i for i in x if i  != 'deleted'])
counts = collections.Counter(x)
counts.most_common(15)

clean_all = pd.DataFrame(counts.most_common(15),
                             columns=['words', 'count'])

fig, ax = plt.subplots(figsize=(8, 8))

# Plot horizontal bar graph
clean_all.sort_values(by='count').plot.barh(x='words',
                      y='count',
                      ax=ax,
                      color="purple")

ax.set_title("Common Words in all Reddit Comments")

plt.show()


#Lets look at the most common words within the top scoring posts
top = df.nlargest(100, 'score', keep = 'all')
top = list(itertools.chain(*top['clean_split']))
top = ([i for i in top if i  != ''])
top = ([i for i in top if i  != 'deleted'])
top_counts = collections.Counter(top)
top_counts.most_common(15)

clean_top = pd.DataFrame(top_counts.most_common(15),
                             columns=['words', 'count'])

fig, ax = plt.subplots(figsize=(8, 8))

# Plot horizontal bar graph
clean_top.sort_values(by='count').plot.barh(x='words',
                      y='count',
                      ax=ax,
                      color="blue")

ax.set_title("Common Words in Top Reddit Comments")

plt.show()





#Bottom scoring

bottom = df.nsmallest(100, 'score', keep = 'all')
bottom = list(itertools.chain(*bottom['clean_split']))
bottom = ([i for i in bottom if i  != ''])
bottom = ([i for i in bottom if i  != 'deleted'])
bottom_counts = collections.Counter(bottom)
bottom_counts.most_common(15)

clean_bottom = pd.DataFrame(bottom_counts.most_common(15),
                             columns=['words', 'count'])

fig, ax = plt.subplots(figsize=(8, 8))

# Plot horizontal bar graph
clean_bottom.sort_values(by='count').plot.barh(x='words',
                      y='count',
                      ax=ax,
                      color="red")

ax.set_title("Common Words in Bottom Reddit Comments")

plt.show()



"""

Look within specific subs

"""


df['clean_split'] = df.body.apply(clean_tweet)

subreddits = df.subreddit.unique()

#This is an attempt to remove outliers

#All!


for x in subreddits:
    df_plot = df.loc[df['subreddit'] == x]
    #Most common words overall
    sub = x
    all_words= list(itertools.chain(*df_plot['clean_split']))
    x = ([i for i in all_words if i  != ''])
    x = ([i for i in x if i  != 'deleted'])
    counts = collections.Counter(x)
    counts.most_common(15)
    
    clean_all = pd.DataFrame(counts.most_common(15),
                                 columns=['words', 'count'])
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot horizontal bar graph
    clean_all.sort_values(by='count').plot.barh(x='words',
                          y='count',
                          ax=ax,
                          color="purple")
    
    ax.set_title("Common Words in all Reddit Comments, Subreddit: " + sub)
    
    plt.show()


#Top!


for x in subreddits:
    df_plot = df.loc[df['subreddit'] == x]
    #Most common words overall
    sub = x
    #Lets look at the most common words within the top scoring posts
    top = df_plot.nlargest(100, 'score', keep = 'all')
    top = list(itertools.chain(*top['clean_split']))
    top = ([i for i in top if i  != ''])
    top = ([i for i in top if i  != 'deleted'])
    top_counts = collections.Counter(top)
    top_counts.most_common(15)
    
    clean_top = pd.DataFrame(top_counts.most_common(15),
                                 columns=['words', 'count'])
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot horizontal bar graph
    clean_top.sort_values(by='count').plot.barh(x='words',
                          y='count',
                          ax=ax,
                          color="blue")
    
    ax.set_title("Common Words in Top Reddit Comments, Subreddit: " + sub)
    
    plt.savefig('C:/Users/Pablo/Documents/Data Incubator/Reddit/New/Test_App_2/test_app_page_2/static/top_freq_' + sub + '.png')
    plt.show()
    

#Bottom!


for x in subreddits:
    df_plot = df.loc[df['subreddit'] == x]
    #Most common words overall
    sub = x
    bottom = df_plot.nsmallest(100, 'score', keep = 'all')
    bottom = list(itertools.chain(*bottom['clean_split']))
    bottom = ([i for i in bottom if i  != ''])
    bottom = ([i for i in bottom if i  != 'deleted'])
    bottom_counts = collections.Counter(bottom)
    bottom_counts.most_common(15)
    
    clean_bottom = pd.DataFrame(bottom_counts.most_common(15),
                                 columns=['words', 'count'])
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot horizontal bar graph
    clean_bottom.sort_values(by='count').plot.barh(x='words',
                          y='count',
                          ax=ax,
                          color="red")
    
    ax.set_title("Common Words in Bottom Reddit Comments, Subreddit: " + sub)
    plt.savefig('C:/Users/Pablo/Documents/Data Incubator/Reddit/New/Test_App_2/test_app_page_2/static/bottom_freq_' + sub + '.png')
    plt.show()
    
