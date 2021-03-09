# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 10:40:53 2020

@author: Pablo
"""
# -*- coding: utf-8 -*-


#This runs very basic models with no tuning on some of my data to help pick the best model





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
import warnings
warnings.filterwarnings('ignore')




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

def plot_residuals(y_test, y_predicted):
    """"
    Plots the distribution for actual and predicted values of the target variable. Also plots the distribution for the residuals
    """
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharey=True)
    sns.distplot(y_test, ax=ax0, kde = False)
    ax0.set(xlabel='Test scores',fontsize=20)
    sns.distplot(y_predicted, ax=ax1, kde = False)
    ax1.set(xlabel="Predicted scores",fontsize=20)
    plt.show()
    fig, ax2 = plt.subplots()
    sns.distplot((y_test-y_predicted), ax = ax2,kde = False)
    ax2.set(xlabel="Residuals",fontsize=20)
    plt.show()
    
    
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



#Read in the Overall DataFrameS
# older data set df = pd.read_csv('C:/Users\Pablo\Documents\Data Challenges\Reddit\Datasets\comments_sent_text.csv')
#df = pd.read_csv('C:/Users\Pablo\Documents\Data Incubator/Reddit/New/Data/df_all_cleaned.csv')
#df = pd.read_csv('C:/Users\Pablo\Documents\Data Incubator/Reddit/New/Data/df_all_cleaned_dropped.csv')
df = pd.read_csv('C:/Users\Pablo\Documents\Data Incubator/Reddit/New/Data/df_all_cleaned_dropped_95.csv')

#use smaller dataset
df = df.sample(n = 300000, random_state=1)


#df_all_dropped = pd.read_csv('C:/Users\Pablo\Documents\Data Incubator/Reddit/New/Data/df_all_cleaned_dropped.csv')
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



"""
#Select the variables we want to use when training the model
#bool_cols = ['is_submitter', 'stickied', 'archived']
bool_cols = ['is_submitter', 'stickied', 'archived', 'christian_text', 'judiasm_text', 'islam_text', 'religion_text',
             'Republican_text', 'Democrat_text', 'Politics_text', 'happy_words', 'sad_words', 'anger_words',
             'Libertarian_text', 'curse_words']
cat_cols=['subreddit', 'submission_title', 'time_hour', 'time_day']
#cat_cols=['subreddit', 'time_hour', 'time_day', 'sentiment1']
numeric_cols = ['parent_score', 'body_length', 'parent_length', 'gilded']
"""


bool_cols = ['is_submitter', 'stickied',  'christian_text', 'judiasm_text', 'islam_text', 'religion_text',
             'Republican_text', 'Democrat_text', 'Politics_text', 'happy_words', 'sad_words', 'anger_words',
             'Libertarian_text', 'curse_words']

cat_cols=['subreddit',  'sentiment1']
numeric_cols = ['parent_score', 'body_length', 'parent_length', 'time_hour', 'time_day','gilded']







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







"""MODELING"""
#Create a dictionary that will store the results of the model
model_performance_dict = dict()

"""Linear Regression Models"""
#Baseline
baseline = DummyRegressor(strategy='mean')
baseline.fit(X_train,y_train)
model_performance_dict["Baseline"] = model_diagnostics(baseline)

#Linear Regression
linear = LinearRegression()
linear.fit(X_train,y_train)
model_performance_dict["Linear Regression"] = model_diagnostics(linear)

#Lasso Regression
lasso = LassoCV(cv=30).fit(X_train, y_train)
model_performance_dict["Lasso Regression"] = model_diagnostics(lasso)

#Ridge Regression
ridge = RidgeCV(cv=10).fit(X_train, y_train)
model_performance_dict["Ridge Regression"] = model_diagnostics(ridge)


"""NonLinear Regression Models"""
#K-Nearest Neighbor Regression
knr = KNeighborsRegressor()
knr.fit(X_train, y_train)
model_performance_dict["KNN Regression"] = model_diagnostics(knr)

#Decision Tree Regression
dt = DecisionTreeRegressor(min_samples_split=45, min_samples_leaf=45, random_state = 10)
dt.fit(X_train, y_train)
model_performance_dict["Decision Tree"] = model_diagnostics(dt)

#Random Forest Regression
rf = RandomForestRegressor(n_jobs=-1, n_estimators=70, min_samples_leaf=10, random_state = 10)
rf.fit(X_train, y_train)
model_performance_dict["Random Forest"] = model_diagnostics(rf)

#Gradient Boosting Regression
gbr = GradientBoostingRegressor(n_estimators=70, max_depth=5)
gbr.fit(X_train, y_train)
model_performance_dict["Gradient Boosting Regression"] = model_diagnostics(gbr)



rf_importances = get_feature_importance(rf)











def model_comparison(model_performance_dict, sort_by = 'RMSE', metric = 'RMSE'):

    Rsq_list = []
    RMSE_list = []
    MAE_list = []
    for key in model_performance_dict.keys():
        Rsq_list.append(model_performance_dict[key][0])
        RMSE_list.append(model_performance_dict[key][1])
        MAE_list.append(model_performance_dict[key][2])

    props = pd.DataFrame([])

    props["R-squared"] = Rsq_list
    props["RMSE"] = RMSE_list
    props["MAE"] = MAE_list
    props.index = model_performance_dict.keys()
    props = props.sort_values(by = sort_by)

    fig, ax = plt.subplots(figsize = (12,6))
    plt.style.use('seaborn')
    ax.bar(props.index, props[metric], color="blue")
    plt.title(metric,fontsize=24)
    plt.xlabel('Model',fontsize=22)
    plt.xticks(rotation = 45, fontsize=16)
    plt.ylabel(metric,fontsize=22)
    
    
model_comparison(model_performance_dict, sort_by = 'R-squared', metric = 'R-squared')    
model_comparison(model_performance_dict, sort_by = 'R-squared', metric = 'MAE')    
model_comparison(model_performance_dict, sort_by = 'R-squared', metric = 'RMSE')    
    
    
#save the model performance dict with pickle

import pickle

with open('model_comparison_dict_alldata_all_features.pickle', 'wb') as handle:
    pickle.dump(model_comparison, handle, protocol=pickle.HIGHEST_PROTOCOL)

#with open('filename.pickle', 'rb') as handle:
    #b = pickle.load(handle)
  
    