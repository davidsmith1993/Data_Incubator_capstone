# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 11:05:31 2021

@author: Pablo
"""

import pandas as pd

df1 = pd.read_csv('C:/Users/Pablo/Documents/Data Incubator/Reddit/New/Data/comments_sent_text.csv')
df2 = pd.read_csv('C:/Users/Pablo\Documents/Data Incubator/Reddit/New/Data/comments_sent_text_1_12_2021.csv')

df_all = df1.append(df2, ignore_index=True)

df_all_small =  df_all.sample(n = 500)

df_all.to_csv(r'C:/Users/Pablo/Documents/Data Incubator/Reddit/New/Data/all.csv') 
df_all_small.to_csv(r'C:/Users/Pablo/Documents/Data Incubator/Reddit/New/Data/all_small.csv') 

#create a cleaned df

cols = [1,2,3]
df_all_small_cleaned =  df_all_small.drop(df_all_small.columns[cols],axis=1)
df_all_cleaned =  df_all.drop(df_all.columns[cols],axis=1)


df_all_small_cleaned.to_csv(r'C:/Users/Pablo/Documents/Data Incubator/Reddit/New/Data/df_all_small_cleaned.csv') 
df_all_cleaned.to_csv(r'C:/Users/Pablo/Documents/Data Incubator/Reddit/New/Data/df_all_cleaned.csv') 




#drop top cases, outliers?
#Drops the top 5 percent of scores
df_all_small_cleaned.sort_values('score', ascending=False, inplace=True)
df_all_small_cleaned_dropped = df_all_small_cleaned[df_all_small_cleaned.score < df_all_small_cleaned.score.quantile(.99)]


df_all_cleaned.sort_values('score', ascending=False, inplace=True)
df_all_cleaned_dropped = df_all_cleaned[df_all_cleaned.score < df_all_cleaned.score.quantile(.99)]
df_all_cleaned_dropped_95 = df_all_cleaned[df_all_cleaned.score < df_all_cleaned.score.quantile(.95)]




df_all_small_cleaned_dropped.to_csv(r'C:/Users/Pablo/Documents/Data Incubator/Reddit/New/Data/df_all_small_cleaned_dropped.csv') 
df_all_cleaned_dropped.to_csv(r'C:/Users/Pablo/Documents/Data Incubator/Reddit/New/Data/df_all_cleaned_dropped.csv') 
df_all_cleaned_dropped_95.to_csv(r'C:/Users/Pablo/Documents/Data Incubator/Reddit/New/Data/df_all_cleaned_dropped_95.csv') 