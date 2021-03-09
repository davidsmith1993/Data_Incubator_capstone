# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 18:18:50 2021

@author: Pablo
"""

#https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74


#import the libraries
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
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')


def get_feature_importance(model):
    """
    For fitted tree based models, get_feature_importance can be used to get the feature importance as a tidy output
    """
    X_non_text = pd.get_dummies(df[cat_cols])
    features = numeric_cols + bool_cols + list(X_non_text.columns)
    feature_importance = dict(zip(features, model.feature_importances_))
    for name, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:<30}: {importance:>6.2%}")
        print(f"\nTotal importance: {sum(feature_importance.values()):.2%}")
    return feature_importance







#df = pd.read_csv('C:/Users\Pablo\Documents\Data Incubator/Reddit/New/Data/df_all_cleaned_dropped_95.csv')
df = pd.read_csv('C:/Users\Pablo\Documents\Data Incubator/Reddit/New/Data/df_all_cleaned_dropped_95_nlp.csv')
df = df.sample(n = 100000, random_state=1)

#df_all_dropped = pd.read_csv('C:/Users\Pablo\Documents\Data Incubator/Reddit/New/Data/df_all_cleaned_dropped.csv')
df['time'] = pd.to_datetime(df['time'])
df['banned_time'] = pd.to_datetime(df['banned_time'])
#Drop NA files
df = df[~df.christian_text.isna()]
df = df[~df.submission_title.isna()] # drop where parent/title_cosine is NaN
df = df[~df.sentiment1.isna()] # drop where parent/title_cosine is NaN
df = df[~df.clean.isna()]
#impute parent score column
parent_scrore_impute = df.parent_score.mode()[0] # impute with mode of parent_score column
df.loc[df.parent_score.isna(), 'parent_score'] = parent_scrore_impute
parent_scrore_impute = df.parent_length.mode()[0] # impute with mode of parent_score column
df.loc[df.parent_length.isna(), 'parent_length'] = parent_scrore_impute







#bool_cols = ['is_submitter', 'stickied', 'archived']
bool_cols = ['is_submitter', 'stickied',  'christian_text', 'judiasm_text', 'islam_text', 'religion_text',
             'Republican_text', 'Democrat_text', 'Politics_text', 'happy_words', 'sad_words', 'anger_words',
             'Libertarian_text', 'curse_words']

cat_cols=['subreddit', 'sentiment1']
numeric_cols = ['parent_score', 'body_length', 'parent_length', 'gilded', 'time_hour', 
                'time_day', 'FleschReadabilityEase', 'upper', 'q_mark']





#Only good features?
bool_cols = ['is_submitter',  
             'Republican_text', 'Politics_text',  'curse_words']
#cat_cols=['subreddit', 'time_hour', 'time_day']
cat_cols=['subreddit', 'sentiment1']
numeric_cols = ['parent_score', 'body_length', 'parent_length', 'gilded', 'time_hour', 
                'time_day', 'FleschReadabilityEase', 'upper']





vectorizer = CountVectorizer(max_features=300, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))

vec_words = vectorizer.fit_transform(df['clean']).toarray()



from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(vec_words).toarray()


#dense_matrix = X.todense()

y = df.score.values
#Split into an 80-20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#Random Forest Regression
rf = RandomForestRegressor(n_jobs=6, n_estimators=85, min_samples_leaf=29, min_samples_split = 16, random_state = 10)
rf.fit(X_train, y_train)
y_predicted = rf.predict(X_test)

rf_importances = get_feature_importance(rf)






#Lets use scikit-leanrs labelbinarizer to make dummy variables
lb = LabelBinarizer()
cat = [lb.fit_transform(df[col]) for col in cat_cols]
bol = [df[col].astype('int') for col in bool_cols]
t = df.loc[:, numeric_cols].values
final = [t] + bol + cat + X
y = df.score.values
x = np.column_stack(tuple(final))





#Split into an 80-20 split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



#Random Forest Regression
rf = RandomForestRegressor(n_jobs=6, n_estimators=85, min_samples_leaf=29, min_samples_split = 16, random_state = 10)
rf.fit(X_train, y_train)
y_predicted = rf.predict(X_test)

rf_importances = get_feature_importance(rf)




import pickle

with open('rf_alldata_all_features.pickle', 'wb') as handle:
    pickle.dump(rf, handle, protocol=pickle.HIGHEST_PROTOCOL)






def y_test_vs_y_predicted(y_test,y_predicted):
    """
    Produces a scatter plot for the actual and predicted values of the target variable
    """
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_predicted)
    ax.set_xlabel("Test Scores")
    ax.set_ylim([-75, 1400])
    ax.set_ylabel("Predicted Scores")
    plt.show()


y_test_vs_y_predicted(y_test, y_predicted)




def plot_residuals(y_test, y_predicted):
    """"
    Plots the distribution for actual and predicted values of the target variable. Also plots the distribution for the residuals
    """
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharey=True)
    sns.distplot(y_test, ax=ax0, kde = False)
    ax0.set(xlabel='Test scores')
    sns.distplot(y_predicted, ax=ax1, kde = False)
    ax1.set(xlabel="Predicted scores")
    plt.show()
    fig, ax2 = plt.subplots()
    sns.distplot((y_test-y_predicted), ax = ax2,kde = False)
    ax2.set(xlabel="Residuals")
    plt.show()
    
plot_residuals(y_test, y_predicted)





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



from sklearn.model_selection import GridSearchCV

"""
est = RandomForestRegressor()
gs = GridSearchCV(
    est,
    {"min_samples_split": range(2,20), "n_estimators" : [85]},  # range of hyperparameters to test
    cv=10,  # 10-fold cross validation
    n_jobs=-1,  # run each hyperparameter in one of two parallel jobs
)
gs.fit(X_train, y_train)

gs.best_params_

#85 is the best? for n_estimator
# "min_samples_leaf" 31

plt.plot(X_train, y_train, 'k.', label='data')
line = plt.plot(X_train, gs.predict(X_train), label='model')
plt.setp(line, linewidth=3., alpha=0.7)
plt.title('Daily Store Sales Given Number of Customers That Day')
plt.xlabel('Number of customers')
plt.ylabel('Store sales [dollars]')
plt.legend(loc='upper left');

"""

len(rf.estimators_)

from sklearn import tree
plt.figure(figsize=(20,20))
_ = tree.plot_tree(rf.estimators_[0], filled=True)

