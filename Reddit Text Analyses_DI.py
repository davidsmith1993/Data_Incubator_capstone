# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:05:22 2020

@author: Pablo
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/Users\Pablo\Documents\Data Challenges\Reddit/Datasets/NLPcomments_NLP_sent_1_12_20.csv')


"""
First real question of the data set
Does religious text predict comment score?
"""
#Lets define some religious text (non extensive of course)
christian_text = ('god|christian|jesus|faith|church|catholic|protestant|evangelical|christ|lord|baptist|orthodox|worship|bible|lutheran')
judiasm_text = ('judaism|jew|isreal|holocaust|hebrew|passover|synagogue|yiddish|jewish|isrealite|hannukah|yom kippur')
islam_text = ('muslim|sunni|islam|islamic|arabic|islamist|muhammad|shiite')
#now, create a variable in the data set for mentions of the text. 
#The text column is the one of interest
df['christian_text'] = df['body'].str.contains(christian_text, case = False)
df['judiasm_text'] = df['body'].str.contains(judiasm_text, case = False)
df['islam_text'] = df['body'].str.contains(islam_text, case = False)
df['religion_text'] = df['body'].str.contains('religion', case = False)



#Look at politics too
Republican_text = ('Donald|Trump|DonaldTrump|Pence|Romney|Reagan|Gop|Polanski|Republican|conservative|Mccain|McConnel')
Democrat_text = ('Clinton|Hillary|Sanders|Yang|Bernie|Sanders|Biden|Obama|Barack|Democrat|Pelosi|Huckabee|Joe')
Libertarian_text = ('Jorgenson|Libertarian|Third party|Jojo')
Politics_text = ('politic|politics|president|government|political|state|politicion|vote|law|governor|court|supreme|justice|police|federal|office|impeach|political party|dictator|legal|illegal|lobby')

df['Republican_text'] = df['body'].str.contains(Republican_text, case = False)
df['Democrat_text'] = df['body'].str.contains(Democrat_text, case = False)
df['Politics_text'] = df['body'].str.contains(Politics_text, case = False)
df['Libertarian_text'] = df['body'].str.contains(Libertarian_text, case = False)



"""The emotion of the words"""

happy_words = ('happy|cheerful|merry|joy|overjoyed|delight|delighted|glad|blessed|wholesom|optimistic')
sad_words = ('sad|bitter|dismal|upset|heartbroken|mourn|sorry|pessimistic|somber|melancholy')
anger_words = ('acrimony|anger|angry|animosity|annoyed|displeasure|fury|hatred|evil|irrate|rage|resent|violence|outrage|mad')

df['happy_words'] = df['body'].str.contains(happy_words, case = False)
df['sad_words'] = df['body'].str.contains(sad_words, case = False)
df['anger_words'] = df['body'].str.contains(anger_words, case = False)



curse_words = ('fuck|shit|bitch|bastard|asshole|damn|cunt')
df['curse_words'] = df['body'].str.contains(curse_words, case = False)
#df['no_curse_words'] = ~df['body'].str.contains(curse_words, case = False)


#df.groupby('curse_words')['score'].agg('mean')



df.to_csv(r'C:/Users/Pablo/Documents/Data Challenges/Reddit/Datasets/Text/comments_sent_text_1_12_2021.csv') 

#try to plot Religion

subreddits = df.subreddit.unique()

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
    
    
    g = sns.barplot(data = dfreligionlong, ci = 60, errwidth=.5
                ,x = 'group'
                ,y = 'score', hue = 'sentiment1',
                )
    plt.title(x + "Average Comment Score by Religion and Text Sentiment")
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

for x in subreddits:
    df_plot = df.loc[df['subreddit'] == 'politics']
    sns.set()
    sns.set_palette("Paired")
    dfreligion = df_plot[['score', 'Republican_text', 'Democrat_text', 'Politics_text', 'Libertarian_text', 'sentiment1']]
    
    dfreligion.loc[dfreligion['Republican_text'] == True, 'Republican_text'] = dfreligion['score']
    dfreligion.loc[dfreligion['Democrat_text'] == True, 'Democrat_text'] = dfreligion['score']
    dfreligion.loc[dfreligion['Politics_text'] == True, 'Politics_text'] = dfreligion['score']
    dfreligion.loc[dfreligion['Libertarian_text'] == True, 'Libertarian_text'] = dfreligion['score']
    
    dfreligion = dfreligion[['Republican_text', 'Democrat_text', 'Libertarian_text', 'Politics_text',  'sentiment1']]
    
    for x in dfreligion['Republican_text']:
        if x > (dfreligion['Republican_text'].mean() * 10):
            x = dfreligion['Republican_text'].mean()
        else:
            x = x
    
    
    
    dfreligionlong = pd.melt(dfreligion, id_vars=['sentiment1'], var_name = 'group',  value_name = 'score')
    dfreligionlong = dfreligionlong[dfreligionlong.score != False]
    dfreligionlong['score'] = dfreligionlong.score.astype(float)
    
    
    g = sns.barplot(data = dfreligionlong, ci = 60, errwidth=.5
                ,x = 'group'
                ,y = 'score', hue = 'sentiment1',
                )
    plt.title(x + " Average Comment Score by Politics and Text Sentiment")
    plt.legend(title = 'Sentiment')
    g.set_xticklabels(['Republican','Democrat','Libertarian', 'General Politics'])
    plt.show()



