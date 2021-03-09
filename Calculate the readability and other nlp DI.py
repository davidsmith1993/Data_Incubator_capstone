# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 09:15:10 2021

@author: Pablo
"""

import pandas as pd
import numpy as np
import statistics

df = pd.read_csv('C:/Users\Pablo\Documents\Data Incubator/Reddit/New/Data/df_all_cleaned_dropped_95.csv')

#df = dfa.sample(n = 9000, random_state=1)

#Flesch Reading Ease (try manually)

def FleschReadabilityEase(text):
	if len(text) > 0:
		return 206.835 - (1.015 * len(text.split()) / len(text.split('.')) ) - 84.6 * (sum(list(map(lambda x: 1 if x in ["a","i","e","o","u","y","A","E","I","O","U","y"] else 0,text))) / len(text.split()))
dfall = pd.DataFrame()    
df2 = pd.DataFrame()
place = 0
for i in df['body']:
    i = str(i)
    score = FleschReadabilityEase(i)
    df2.loc[place,'FleschReadabilityEase'] = score
    place = place + 1


df['FleschReadabilityEase'] = df2['FleschReadabilityEase'].values

    
np.corrcoef(df['FleschReadabilityEase'], df['score'])





#pip install spacy

#spacy.load("en_core_web_sm")
#pip install spacy-readability
#pip install spacy_readability


import spacy
from spacy_readability import Readability
nlp = spacy.load('en_core_web_sm')
read = Readability(nlp)


y = Counter(([token.pos_ for token in nlp('The cat sat on the mat.')]))

y['NOUN']/len([token.pos_ for token in nlp('The cat sat on the mat.')])



dfall = pd.DataFrame()    
df2 = pd.DataFrame()
place = 0
for i in df['body']:
    i = str(i)
    y = Counter(([token.pos_ for token in nlp(i)]))
    noun = y['NOUN']/len([token.pos_ for token in nlp(i)])
    df2.loc[place,'noun'] = noun
    place = place + 1


df['noun'] = df2['noun'].values 

np.corrcoef(df['noun'], df['score'])





#These are what I am actually using

import textstat


#readability
textstat.flesch_reading_ease(test_data)
textstat.text_standard(test_data, float_output=True)

dfall = pd.DataFrame()    
df2 = pd.DataFrame()
place = 0
for i in df['body']:
    i = str(i)
    score = textstat.text_standard(i, float_output=True)
    df2.loc[place,'FleschReadabilityEase'] = score
    place = place + 1


df['FleschReadabilityEase'] = df2['FleschReadabilityEase'].values

    
np.corrcoef(df['FleschReadabilityEase'], df['score'])



df.to_csv(r'C:/Users/Pablo/Documents/Data Incubator/Reddit/New/Data/df_all_cleaned_dropped_95_nlp.csv') 



#Uppercase Ratio
statistics.mean(ch.isupper() for ch in test_data if not ch.isspace())

dfall = pd.DataFrame()    
df2 = pd.DataFrame()
place = 0
for i in df['body']:
    i = str(i)
    upper = statistics.mean(ch.isupper() for ch in i if not ch.isspace())
    df2.loc[place,'upper'] = upper
    place = place + 1


df['upper'] = df2['upper'].values

    
np.corrcoef(df['upper'], df['score'])



#Number of Questions
len(test_data) - len(test_data.rstrip('?'))

dfall = pd.DataFrame()    
df2 = pd.DataFrame()
place = 0
for i in df['body']:
    i = str(i)
    q_mark = len(i) - len(i.rstrip('?'))
    df2.loc[place,'q_mark'] = q_mark
    place = place + 1


df['q_mark'] = df2['q_mark'].values

np.corrcoef(df['q_mark'], df['score'])


#Title is Question
#[:-1]






df.to_csv(r'C:/Users/Pablo/Documents/Data Incubator/Reddit/New/Data/df_all_cleaned_dropped_95_nlp.csv') 