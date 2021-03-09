# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 12:02:30 2021

@author: Pablo
"""
"""

    This creates the model I deployed on my website 
    
"""
import statistics
import textstat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
nltk.download('punkt')


#First we fit the model based on these features



#Get data
df = pd.read_csv('C:/Users\Pablo\Documents\Data Incubator/Reddit/New/Data/df_all_cleaned_dropped_95_nlp.csv')
df['time'] = pd.to_datetime(df['time'])
df['banned_time'] = pd.to_datetime(df['banned_time'])
#Drop NA files
df = df[~df.christian_text.isna()]
df = df[~df.submission_title.isna()] # drop where parent/title_cosine is NaN
df = df[~df.sentiment1.isna()] # drop where parent/title_cosine is NaN
#impute parent score column
parent_scrore_impute = df.parent_score.mode()[0] # impute with mode of parent_score column
df.loc[df.parent_score.isna(), 'parent_score'] = parent_scrore_impute
parent_scrore_impute = df.parent_length.mode()[0] # impute with mode of parent_score column
df.loc[df.parent_length.isna(), 'parent_length'] = parent_scrore_impute

#features we want
bool_cols = []
#cat_cols=['subreddit', 'time_hour', 'time_day']
cat_cols=['sentiment1']
numeric_cols = ['parent_score', 'body_length', 'gilded', 'time_day','time_hour', 'FleschReadabilityEase', 'upper', 'q_mark']

#Lets use scikit-leanrs labelbinarizer to make dummy variables
lb = LabelBinarizer()
cat = [lb.fit_transform(df[col]) for col in cat_cols]
bol = [df[col].astype('int') for col in bool_cols]
t = df.loc[:, numeric_cols].values
final = [t] + bol + cat
y = df.score.values
x = np.column_stack(tuple(final))

#Split into an 80-20 split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Random Forest Regression
rf = RandomForestRegressor(n_jobs=-1, n_estimators=10, min_samples_leaf=29, min_samples_split = 16, random_state = 10)
rf.fit(X_train, y_train)
y_predicted = rf.predict(X_test)

#rf is fit

"""Then we test it with defined features"""

comment = 'I have a large dog and I love him'

body_len = len(comment)

time_h = 1
time_d = 2
gild = 0
par_score = 500


#Our goal here is to conduct sentiment analyses of the comments
sid = SentimentIntensityAnalyzer()


scores = sid.polarity_scores(comment)
sentiment = []


if scores["compound"] >= 0.05:
    sentiment = 'positive'
    s = [0, 0, 1]
    
elif scores["compound"] <= -0.05:
    sentiment = 'negative'
    s = [1, 0 ,0 ]
else:
    sentiment = 'neutral'
    s = [0 , 1 ,0 ]


    read = textstat.text_standard(comment, float_output=True)
    upper = statistics.mean(ch.isupper() for ch in comment if not ch.isspace())
    q_mark = len(comment) - len(comment.rstrip('?'))



c_test = [[par_score, body_len, gild, time_d, time_h, read, upper, q_mark] + s]

c_pred= rf.predict(c_test)
c_pred

def model_diagnostics(model, pr=True):
    """
    Returns and prints the R-squared, RMSE and the MAE for a trained model
    """
    y_predicted = model.predict(X_test)
    r2 = r2_score(y_test, y_predicted)
    mse = mean_squared_error(y_test, y_predicted)
    mae = mean_absolute_error(y_test, y_predicted)
    if pr:
        print(f"R-Sq: {r2:.4}")
        print(f"RMSE: {np.sqrt(mse)}")
        print(f"MAE: {mae}")
    
    return [r2,np.sqrt(mse),mae]


model_diagnostics(rf)















#This saves the model

model = rf

import pickle##dump the model into a file
with open("model.bin", 'wb') as f_out:
    pickle.dump(model, f_out) # write final_model in .bin file
    f_out.close()  # close the file 
    
    
    
#This opens
##loading the model from the saved file
with open('model.bin', 'rb') as f_in:
    model = pickle.load(f_in)    
    
    

