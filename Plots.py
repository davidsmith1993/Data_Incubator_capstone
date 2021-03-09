# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 14:47:29 2020

@author: dsmit
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#try to plot Religion


df_all = pd.read_csv('C:/Users\Pablo\Documents\Data Incubator/Reddit/New/Data/df_all_cleaned.csv')
df_all_dropped = pd.read_csv('C:/Users\Pablo\Documents\Data Incubator/Reddit/New/Data/df_all_cleaned_dropped.csv')
df_all_dropped_95 = pd.read_csv('C:/Users\Pablo\Documents\Data Incubator/Reddit/New/Data/df_all_cleaned_dropped_95.csv')


df_all_cleaned_dropped_95_05 = df_all_dropped_95[df_all_dropped_95.score > df_all_dropped_95.score.quantile(.01)]

df = df_all_cleaned_dropped_95_05


#subreddits = df.subreddit.unique()
subreddits = ['politics']


for x in subreddits:
    df_plot = df.loc[df['subreddit'] == x]
    sns.set()
    sns.set_palette("Paired")
    dfreligion = df_plot[['score', 'christian_text', 'judiasm_text', 'islam_text', 'religion_text', 'sentiment1']]
    
    dfreligion.loc[dfreligion['christian_text'] == True, 'christian_text'] = dfreligion['score']
    dfreligion.loc[dfreligion['judiasm_text'] == True, 'judiasm_text'] = dfreligion['score']
    dfreligion.loc[dfreligion['islam_text'] == True, 'islam_text'] = dfreligion['score']
    dfreligion.loc[dfreligion['religion_text'] == True, 'religion_text'] = dfreligion['score']
    
    dfreligion = dfreligion[['christian_text', 'judiasm_text', 'islam_text', 'religion_text', 'sentiment1']]
    dfreligionlong = pd.melt(dfreligion, id_vars=['sentiment1'], var_name = 'group',  value_name = 'score')
    dfreligionlong = dfreligionlong[dfreligionlong.score != False]
    dfreligionlong['score'] = dfreligionlong.score.astype(float)
    
    #dfreligionlong2 = dfreligionlong[(dfreligionlong['score']>5) & (dfreligionlong['score']< 100)]
    
    g = sns.barplot(data = dfreligionlong, ci = 60, errwidth=.5
                ,x = 'group'
                ,y = 'score', hue = 'sentiment1', palette="Blues_d", 
                )
    plt.title(" Average Comment Score by Religion and Text Sentiment, Subreddit: " + x)
    plt.legend(title = 'Sentiment')
    g.set_xticklabels(['Christianity','Judaism','Islam', 'General Religion'])
    plt.show()
    plt.clf()







sns.set()
sns.set_palette("Paired")

dfcurse = df[['score', 'curse_words', 'no_curse_words', 'sentiment1']]

dfcurse.loc[dfcurse['curse_words'] == True, 'curse_words'] = dfcurse['score']
dfcurse.loc[dfcurse['no_curse_words'] == True, 'no_curse_words'] = dfcurse['score']


dfcurse = dfcurse[['curse_words', 'no_curse_words','sentiment1']]
dfcurselong = pd.melt(dfcurse, id_vars=['sentiment1'], var_name = 'group',  value_name = 'score')
dfcurselong = dfcurselong[dfcurselong.score != False]
dfcurselong['score'] = dfcurselong.score.astype(float)


g = sns.barplot(data = dfcurselong, ci = 60, errwidth=.5
            ,x = 'group'
            ,y = 'score', hue = 'sentiment1',
            )
plt.title("Average Comment Score by Inclusion of Curse Words and Text Sentiment")
plt.legend(title = 'Sentiment')
g.set_xticklabels(['Curse','No Curse'])








sns.set()
sns.set_palette("Paired")
dfreligion = df[['score', 'happy_words', 'sad_words', 'anger_words', 'sentiment1']]

dfreligion.loc[dfreligion['happy_words'] == True, 'happy_words'] = dfreligion['score']
dfreligion.loc[dfreligion['sad_words'] == True, 'sad_words'] = dfreligion['score']
dfreligion.loc[dfreligion['anger_words'] == True, 'anger_words'] = dfreligion['score']


dfreligion = dfreligion[['happy_words', 'sad_words', 'anger_words',  'sentiment1']]
dfreligionlong = pd.melt(dfreligion, id_vars=['sentiment1'], var_name = 'group',  value_name = 'score')
dfreligionlong = dfreligionlong[dfreligionlong.score != False]
dfreligionlong['score'] = dfreligionlong.score.astype(float)


g = sns.barplot(data = dfreligionlong, ci = 60, errwidth=.5
            ,x = 'group'
            ,y = 'score', hue = 'sentiment1',
            )
plt.title("Average Comment Score by Religion and Text Sentiment")
plt.legend(title = 'Sentiment')
g.set_xticklabels(['Happy','Sad','Angry'])


subreddits = df.subreddit.unique()

