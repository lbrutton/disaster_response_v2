## Classifying social media messages related to disaster response

As part of a data science and machine learning course, built this webapp and optimized machine learning algorithm to help classify social media messages, based on which disaster response department the message related to.

## Contents of the repository

- webapp: contains the full web application, which can be deployed locally and contains scripts to generate the machine learning models
- DisasterResponse.db: SQLite database containing the main data model for this exercise, taken from the different csv files and piped using the steps outlined in the ETL Pipeline Preparation file
- ETL Pipeline Preparation.ipynb: jupyter notebook which breaks down how we prepare and model the data in preparation for this exercise
- ML Pipeline Preparation.ipynb: jupyter notebook detailing the data preprocessing and model selection and optimization for the machine learning
- Categories.csv: raw data used for this exercise, contains the different categories we want to classify our messages into
- messages.csv: raw data used for this exercise, contain the actual social media messages that will be processed

## Business questions

- What is the best model to help classify social media messages in times of natural disasters, based on which emergency department is best suited to respond?

## Conclusions

We found that a Random Forest model with 100 estimators worked best out of all the models we tested. However there is definitely room for improvement here, potentially using a different system of document vectorization, such as Latent Dirichlet Allocation.

## Libraries used:

from sqlalchemy import create_engine
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import pdb
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import pickle
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize 
import re
import string
