# Data_Incubator_capstone
This is a project completed for The Data Incubator. The final form of the project can be seen on davidsmith-capstone.herokuapp.com
The following scripts were used to gather the data, clean the data, process the data, and create the figures for my website.


## **Reddit Pull Comments_Reddit_API.py**
Here we are using Praw to gather comments from the social media website Reddit.com from a list of popular subreddits.


## **Base NLP_DI.py**
Here we clean the text data from the comments and conduct sentiment analyses of the comments.

## **Calculate the readability.py**
The reading level of the text, as well as some other information, is extracted from the text.

## Combine DataSets.py
Simple script to combine datasets into one dataset and remove extreme scoring comments.

## Reddit base models.py
A number of models were run on some features from the comments themselves. A random forest regression was selected as the best scoring model.

## Random Forest Regression DI.py
Final model created to predict comment score. More information can be seen st https://davidsmith-capstone.herokuapp.com/model_fits


# Plots
## Plots.py
Generates plots showing score within subreddits by religion and political affiliation
## Word Freq.py
Plots showing the most frequent words for top and bottom scoring comments

## Histogram of Score.py 
Generate Basic plots looking at the distribution on the score

## Deployable model.py 
The random forest regression model used at https://davidsmith-capstone.herokuapp.com/deployed
