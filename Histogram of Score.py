# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 10:56:19 2021

@author: Pablo
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



df_all = pd.read_csv('C:/Users\Pablo\Documents\Data Incubator/Reddit/New/Data/df_all_cleaned.csv')
df_all_dropped = pd.read_csv('C:/Users\Pablo\Documents\Data Incubator/Reddit/New/Data/df_all_cleaned_dropped.csv')
df_all_dropped_95 = pd.read_csv('C:/Users\Pablo\Documents\Data Incubator/Reddit/New/Data/df_all_cleaned_dropped_95.csv')



sns.set_color_codes()
sns.set_style("white")

#For all the data
plt.figure()
#plt.xlim(0,2000)

plt.ylabel('Count of Comments')
plt.title('Histogram of Comment Score for All Scores')
sns.distplot(df_all['score'],bins=150,kde=False, color= 'k')
#sns.kdeplot(df_all['score'],shade=True) 
plt.xlabel('Comment Score')
plt.show()

plt.figure(figsize=(10,5))
#plt.xlim(0,1000)
sns.boxplot(df_all['score'])
plt.show()


#For bottom 99% of scores
plt.figure()
#plt.xlim(0,2000)

plt.ylabel('Count of Comments')
plt.title('Histogram of Comment Score for the Bottom 99% of Scores')
sns.distplot(df_all_dropped['score'],bins=150,kde=False, color= 'k')
plt.xlabel('Comment Score')
#sns.kdeplot(df_all_dropped['score'],shade=True) 
plt.show()


plt.figure(figsize=(10,5))
#plt.xlim(0,1000)
sns.boxplot(df_all_dropped['score'])
plt.show()


#For bottom 95% of scores
plt.figure()
#plt.xlim(0,2000)
plt.ylabel('Count of Comments')
plt.title('Histogram of Comment Score for the Bottom 95% of Scores')
sns.distplot(df_all_dropped_95['score'],bins=150,kde=False, color= 'k')
#sns.kdeplot(df_all_dropped['score'],shade=True) 
plt.xlabel('Comment Score')
plt.show()


plt.figure(figsize=(10,5))
#plt.xlim(0,1000)
sns.boxplot(df_all_dropped_95['score'])
plt.show()



df_all_cleaned_dropped_95_05 = df_all_dropped_95[df_all_dropped_95.score > df_all_dropped_95.score.quantile(.01)]

#For bottom 95% of scores
plt.figure()
plt.ylim(0,99000)

plt.ylabel('Count of Comments')
plt.title('Histogram of Comment Score for the Bottom 95% of Scores')
sns.distplot(df_all_cleaned_dropped_95_05['score'],bins=150,kde=False, color= 'k')
plt.xlabel('Comment Score')
#sns.kdeplot(df_all_dropped['score'],shade=True) 
plt.show()


plt.figure(figsize=(10,5))
#plt.xlim(0,1000)
sns.boxplot(df_all_cleaned_dropped_95_05['score'])
plt.show()













"""log transformed"""

df_all['log10_score'] = np.log10(df_all['score'] + 1058)



sns.set_color_codes()
sns.set_style("white")

#For all the data
plt.figure()
#plt.xlim(0,2000)

plt.ylabel('Count of Comments')
plt.title('Histogram of Comment Score for All Scores')
sns.distplot(df_all['log10_score'],bins=150,kde=False, color= 'k')
#sns.kdeplot(df_all['score'],shade=True) 
plt.xlabel('Comment Score')
plt.show()









