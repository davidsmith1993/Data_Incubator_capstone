# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 13:21:52 2020

@author: dsmit
"""
#conda config --append channels conda-forge
#conda install praw

import pandas as pd
import praw

reddit = praw.Reddit(client_id="2srEoGpH-ElTzw",
                     client_secret="iAoRJCVrTvCRQD90MBRPRW0UQzg",
                     user_agent="Python data scrap for analyses (by /u/removed",
                     username="removed",
                     password="removed")

"""
Subreddit Lists

conservative subreddits
 -conservative 
liberal subreddits
 -politics -latestagecapitalism
libertarian
 -libertarian


Christian subreddits
-christianity
Islam subreddit
-islam
Judaism
-Judiasm
atheist subreddit
-atheism 

news
-news -worldnews

not politics, not religion
-askreddit -gaming -funny -aww -science

"""


#I need to do this in steps and save along the way since it takes so long


"""politics"""

comments=[]
subreddit_list = ['conservative', 'politics', 'latestagecapitalism', 'libertarian']
for subreddits in subreddit_list:
    hot = reddit.subreddit(subreddits).top(limit=50, time_filter='month')
    sub = subreddits
    for submission in hot:
        #print(submission.title)
        submission.comments.replace_more(limit=10)
        for comment in submission.comments.list():
            if type(comment.parent()) == praw.models.reddit.submission.Submission:
                comment.parent().body = 'NaN'
                comment.parent().score = 'NaN'
            comments.append([sub, submission.title, comment.author, comment.score, comment.body, comment.created_utc, comment.parent().body, comment.parent().score,
                             comment.distinguished, comment.edited, comment.is_submitter, comment.stickied, comment.gilded, comment.archived, comment.banned_at_utc,
                             comment.parent().created_utc])
        
        
        
comments = pd.DataFrame(comments,columns=['subreddit','submission_title', 'author', 'score', 'body', 'time', 'parent_comment_body', 'parent_score',
                                          'distinguished', 'edited', 'is_submitter', 'stickied', 'gilded', 'archived', 'banned_time', 'parent_created_time'])

comments['body_length'] = comments['body'].str.len()
comments['parent_length'] = comments['parent_comment_body'].str.len()

comments['time'] = pd.to_datetime(comments['time'],unit='s')
comments['time_hour'] = comments['time'].dt.hour
comments['time_day'] = comments['time'].dt.day
comments['parent_created_time'] = pd.to_datetime(comments['parent_created_time'],unit='s')
comments.to_csv(r'C:/Users/Pablo/Documents/Data Challenges/Reddit/Datasets/Raw/politics_comments_DI_1_12_20.csv') 



"""religion"""

comments=[]
subreddit_list = ['christianity', 'islam', 'Judiasm', 'atheism']
for subreddits in subreddit_list:
    hot = reddit.subreddit(subreddits).top(limit=50, time_filter='month')
    sub = subreddits
    for submission in hot:
        #print(submission.title)
        submission.comments.replace_more(limit=10)
        for comment in submission.comments.list():
            if type(comment.parent()) == praw.models.reddit.submission.Submission:
                comment.parent().body = 'NaN'
                comment.parent().score = 'NaN'
            comments.append([sub, submission.title, comment.author, comment.score, comment.body, comment.created_utc, comment.parent().body, comment.parent().score,
                             comment.distinguished, comment.edited, comment.is_submitter, comment.stickied, comment.gilded, comment.archived, comment.banned_at_utc,
                             comment.parent().created_utc])
        
        
        
comments = pd.DataFrame(comments,columns=['subreddit','submission_title', 'author', 'score', 'body', 'time', 'parent_comment_body', 'parent_score',
                                          'distinguished', 'edited', 'is_submitter', 'stickied', 'gilded', 'archived', 'banned_time', 'parent_created_time'])

comments['body_length'] = comments['body'].str.len()
comments['parent_length'] = comments['parent_comment_body'].str.len()

comments['time'] = pd.to_datetime(comments['time'],unit='s')
comments['time_hour'] = comments['time'].dt.hour
comments['time_day'] = comments['time'].dt.day
comments['parent_created_time'] = pd.to_datetime(comments['parent_created_time'],unit='s')
comments.to_csv(r'C:/Users/Pablo/Documents/Data Challenges/Reddit/Datasets/Raw/religion_comments_DI_1_12_20.csv') 


""" Other"""
comments=[]
subreddit_list = ['gaming', 'funny', 'aww', 'science']
#subreddit_list = ['news', 'worldnews', 'askreddit', 'gaming', 'funny', 'aww', 'science']
for subreddits in subreddit_list:
    hot = reddit.subreddit(subreddits).top(limit=50, time_filter='month')
    sub = subreddits
    for submission in hot:
        #print(submission.title)
        submission.comments.replace_more(limit=10)
        for comment in submission.comments.list():
            if type(comment.parent()) == praw.models.reddit.submission.Submission:
                comment.parent().body = 'NaN'
                comment.parent().score = 'NaN'
            comments.append([sub, submission.title, comment.author, comment.score, comment.body, comment.created_utc, comment.parent().body, comment.parent().score,
                             comment.distinguished, comment.edited, comment.is_submitter, comment.stickied, comment.gilded, comment.archived, comment.banned_at_utc,
                             comment.parent().created_utc])
        
        
        
comments = pd.DataFrame(comments,columns=['subreddit','submission_title', 'author', 'score', 'body', 'time', 'parent_comment_body', 'parent_score',
                                          'distinguished', 'edited', 'is_submitter', 'stickied', 'gilded', 'archived', 'banned_time', 'parent_created_time'])

comments['body_length'] = comments['body'].str.len()
comments['parent_length'] = comments['parent_comment_body'].str.len()

comments['time'] = pd.to_datetime(comments['time'],unit='s')
comments['time_hour'] = comments['time'].dt.hour
comments['time_day'] = comments['time'].dt.day
comments['parent_created_time'] = pd.to_datetime(comments['parent_created_time'],unit='s')
comments.to_csv(r'C:/Users/Pablo/Documents/Data Challenges/Reddit/Datasets/Raw/otherandnews_comments_DI.csv_1_12_20')

""" Other2"""
comments=[]
subreddit_list = ['news', 'worldnews']
#subreddit_list = ['news', 'worldnews', 'askreddit', 'gaming', 'funny', 'aww', 'science']
for subreddits in subreddit_list:
    hot = reddit.subreddit(subreddits).top(limit=50, time_filter='month')
    sub = subreddits
    for submission in hot:
        #print(submission.title)
        submission.comments.replace_more(limit=10)
        for comment in submission.comments.list():
            if type(comment.parent()) == praw.models.reddit.submission.Submission:
                comment.parent().body = 'NaN'
                comment.parent().score = 'NaN'
            comments.append([sub, submission.title, comment.author, comment.score, comment.body, comment.created_utc, comment.parent().body, comment.parent().score,
                             comment.distinguished, comment.edited, comment.is_submitter, comment.stickied, comment.gilded, comment.archived, comment.banned_at_utc,
                             comment.parent().created_utc])
        
        
        
comments = pd.DataFrame(comments,columns=['subreddit','submission_title', 'author', 'score', 'body', 'time', 'parent_comment_body', 'parent_score',
                                          'distinguished', 'edited', 'is_submitter', 'stickied', 'gilded', 'archived', 'banned_time', 'parent_created_time'])

comments['body_length'] = comments['body'].str.len()
comments['parent_length'] = comments['parent_comment_body'].str.len()

comments['time'] = pd.to_datetime(comments['time'],unit='s')
comments['time_hour'] = comments['time'].dt.hour
comments['time_day'] = comments['time'].dt.day
comments['parent_created_time'] = pd.to_datetime(comments['parent_created_time'],unit='s')
comments.to_csv(r'C:/Users/Pablo/Documents/Data Challenges/Reddit/Datasets/Raw/otherandnews_comments_DI2.csv_1_12_20')

""" Other3"""
comments=[]
subreddit_list = ['sports', 'askmen', 'europe', 'askwomen']
#subreddit_list = ['news', 'worldnews', 'askreddit', 'gaming', 'funny', 'aww', 'science']
for subreddits in subreddit_list:
    hot = reddit.subreddit(subreddits).top(limit=50, time_filter='month')
    sub = subreddits
    for submission in hot:
        #print(submission.title)
        submission.comments.replace_more(limit=10)
        for comment in submission.comments.list():
            if type(comment.parent()) == praw.models.reddit.submission.Submission:
                comment.parent().body = 'NaN'
                comment.parent().score = 'NaN'
            comments.append([sub, submission.title, comment.author, comment.score, comment.body, comment.created_utc, comment.parent().body, comment.parent().score,
                             comment.distinguished, comment.edited, comment.is_submitter, comment.stickied, comment.gilded, comment.archived, comment.banned_at_utc,
                             comment.parent().created_utc])
        
        
        
comments = pd.DataFrame(comments,columns=['subreddit','submission_title', 'author', 'score', 'body', 'time', 'parent_comment_body', 'parent_score',
                                          'distinguished', 'edited', 'is_submitter', 'stickied', 'gilded', 'archived', 'banned_time', 'parent_created_time'])

comments['body_length'] = comments['body'].str.len()
comments['parent_length'] = comments['parent_comment_body'].str.len()

comments['time'] = pd.to_datetime(comments['time'],unit='s')
comments['time_hour'] = comments['time'].dt.hour
comments['time_day'] = comments['time'].dt.day
comments['parent_created_time'] = pd.to_datetime(comments['parent_created_time'],unit='s')
comments.to_csv(r'C:/Users/Pablo/Documents/Data Challenges/Reddit/Datasets/Raw/otherandnews_comments_DI3.csv_1_12_20')

