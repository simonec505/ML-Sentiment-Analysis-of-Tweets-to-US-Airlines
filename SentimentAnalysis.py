# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 19:58:25 2017
Purpose: Given a set of tweets with pre-defined labels, use 3 Supervised Learning algorithms to 
predict the sentiment of tweets that were directed at US airlines. 
Compare the accuracy of the three algorithms.
@author: Simone
"""
#------------------------------------------------------------------------------Import the dataset and the libraries

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Get the data
dataset = pd.read_csv('Tweets.csv')


#------------------------------------------------------------------------------Explore the data
     
#Explore the data - result shows data is skewed (more negative tweets in general).
sentiment_counts = dataset.airline_sentiment.value_counts()
number_of_tweets = dataset.tweet_id.count()
print(sentiment_counts)


#------------------------------------------------------------------------------Clean the data

#Filter out our 'Neutral' results - arguably, these do not contain sentiment.
dataset_new = dataset[(dataset.airline_sentiment == "negative") | (dataset.airline_sentiment == "positive")]
X = dataset_new.iloc[:, 10].values 

# Clean the tweets (remove punctuation; remove tweets starting with @; get word stem)
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
corpus = []
for i in range(0, 11541):
    tweet = X[i]
    tweet = tweet.lower()
    tweet = tweet.split()
    tweet = [word for word in tweet if not word.startswith("@")]
    tweet = ' '.join(tweet)
    tweet = re.sub('[^a-zA-Z]', ' ', tweet)
    tweet = tweet.split()
    ss = SnowballStemmer('english')
    tweet = [ss.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    corpus.append(tweet)

# Create the Bag of Words for the independent variable
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 500) #initially 15000+ returned.
X = cv.fit_transform(corpus).toarray()

# Get the dependent var.
y = dataset_new.iloc[:, 1].values
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#------------------------------------------------------------------------------Model Training and Testing
 
# Split the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
bayes_classifier = GaussianNB()
bayes_classifier.fit(X_train, y_train)

# Fitting the Decision Tree model to the Training set
from sklearn.tree import DecisionTreeClassifier
dec_tree_classifier = DecisionTreeClassifier()
dec_tree_classifier.fit(X_train, y_train)

# Fitting the Random Forest model to the Training set
from sklearn.ensemble import RandomForestClassifier
ranforest_classifier = RandomForestClassifier()
ranforest_classifier.fit(X_train, y_train)

# Predicting the Test set results for different models
y_pred_bayes = bayes_classifier.predict(X_test)
y_pred_dec_tree = dec_tree_classifier.predict(X_test)
y_pred_ranforest = ranforest_classifier.predict(X_test)

# Making the Confusion Matrix for different models
from sklearn.metrics import confusion_matrix
bayes_cm = confusion_matrix(y_test, y_pred_bayes)
dec_tree_cm = confusion_matrix(y_test, y_pred_dec_tree)
ranforest_cm = confusion_matrix(y_test, y_pred_ranforest)

#------------------------------------------------------------------------------Evaluation:

#My Own from CM:
bayes_negative_accuracy = bayes_cm[0][0]/(bayes_cm[0][0]+bayes_cm[0][1])*100
print("Bayes Negative Accuracy: " + str(bayes_negative_accuracy))

bayes_positive_accuracy = bayes_cm[1][1]/(bayes_cm[1][0]+bayes_cm[1][1])*100
print("Bayes Positive Accuracy: " + str(bayes_positive_accuracy))

dec_tree_negative_accuracy = dec_tree_cm[0][0]/(dec_tree_cm[0][0]+dec_tree_cm[0][1])*100
print("Decision Tree Negative Accuracy: " + str(dec_tree_negative_accuracy))

dec_tree_positive_accuracy = dec_tree_cm[1][1]/(dec_tree_cm[1][0]+dec_tree_cm[1][1])*100
print("Decision Tree Positive Accuracy: " + str(dec_tree_positive_accuracy))

ranforest_negative_accuracy = ranforest_cm[0][0]/(ranforest_cm[0][0]+ranforest_cm[0][1])*100
print("Random Forest Negative Accuracy: " + str(ranforest_negative_accuracy))

ranforest_positive_accuracy = bayes_cm[1][1]/(ranforest_cm[1][0]+ranforest_cm[1][1])*100
print("Random Forest Positive Accuracy: " + str(ranforest_positive_accuracy))

#Classification Report

from sklearn.metrics import classification_report
bayes_report = classification_report(y_test, y_pred_bayes)
print(bayes_report)

dec_tree_report = classification_report(y_test, y_pred_dec_tree)
print(dec_tree_report)

ranforest_report = classification_report(y_test, y_pred_ranforest)
print(ranforest_report)

#Area under ROC curve (bc in non-bell curve shaped data, accuracy is not always a good evaluator.) - closer to 1 is better; closer to 0.5 is worse.
from sklearn import model_selection
kfold = model_selection.KFold(n_splits=10, random_state=0)
model = bayes_classifier
scoring = 'roc_auc'
results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
print(results.mean(), results.std())

## Feature engineering.
## Doen selfde testing op training set (in eval seksie). Strategies for overfitting/underfitting.
## Parameters of models. 
## Dummy classifier - sklearn -- compare with this to see if your model does better than it. 
## ROC curve. -- visual depiction
## check test and training set -- biases.-- sklearn -- unbalanced sets. If 40% neg in test, then 40% neg in training. 
## sklearn research - gaan terug en kyk watter wooorde het die meeste impak. 









































"""Getting the dependent var. I thought more nuance might be built into the 
model if, instead of categorical values, continuous values (based on confidence)
 are catered for. I am worried, however, about the confidenc level of neutral
 statements.
 
y = []
for i in range(0, 14640):
    if dataset['airline_sentiment'][i]=="negative":
        y.append(dataset['airline_sentiment_confidence'][i]*-1)
    elif dataset['airline_sentiment'][i]=="positive":
        y.append(dataset['airline_sentiment_confidence'][i])
    else:
        y.append(0)
        
        """





"""
#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

"""
"""
#Importing the dataset
dataset = pd.read_csv("Tweets.csv")
Y = dataset.iloc[:, 1].values
x = dataset.iloc[:, 10].values
Y = Y.tolist()
x = x.tolist()

#Clean independent var (tweets) by getting stem of words (user for later count)
#from nltk.stem.snowball import SnowballStemmer
#stemmer = SnowballStemmer("english")
#x['stemmed'] = x.document.map(lambda x: ' '.join([stemmer.stem(y) for y in x.decode('utf-8').split(' ')]))
#x = stemmer.fit_transform()

#encoding categorical data (dependent var)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#Apply 'bag of words' frequency counting algorithm to the independent var

from sklearn.feature_extraction.text import CountVectorizer 
vectorizer = CountVectorizer(stop_words='english')
print( vectorizer.fit_transform(x).todense() )
print( vectorizer.vocabulary_ )

'''
#Clean independent var (tweets) + count frequency of words
from sklearn.feature_extraction.text import CountVectorizer
import nltk
countvec = CountVectorizer(min_df=1, stop_words='english', tokenizer=nltk.word_tokenize)
x_count = countvec.fit_transform(x)
result = list(map(lambda row:dict(zip(x_count,row)), x_count.to_array()))

print(x_count)
'''

#Splitting the dataset into the training set and the test set
from sklearn.cross_validation import train_test_split
x_train, x_test, Y_train, Y_test = train_test_split(x,Y,test_size=0.2, random_state=0)

#


#taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0) #how should missing vals be dealt with
imputer = imputer.fit(X[:, 1:3])                                 #actually tell it which columns and rows to fit it to
X[:, 1:3] = imputer.transform(X[:, 1:3])                         #apply operation to our X table.

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
onehotencoder = OneHotEncoder(categorical_features=[0])
X[:, 0] = labelencoder_X.fit_transform(X[:,0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder()                                 #fn we're going to use
Y = labelencoder_Y.fit_transform(Y)                     #fit/apply the fn to the dataset

#Splitting the dataset into the training set and the test set. 
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""
