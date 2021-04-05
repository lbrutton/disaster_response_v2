# import libraries
import sys
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


def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', con=engine)
    df = df[df['related'] != 2]
    X = df['message']
    y = df.iloc[:,-36:]
    return X,y
    pass


def tokenize(text):
    """Clean and tokenize words in a given message."""
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    pass


def build_model():
    pipeline_knn = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    return pipeline_knn
    pass

def classify_model_output(y_test, y_pred):
    classification_scores = []

    for i, column in enumerate(y_test.columns):
        # print(column + str(i) + ' has the following score: \n' + classification_report(y_test[column], y_pred[:, i]))
        classification_scores.append(precision_recall_fscore_support(y_test[column], y_pred[:, i]))
        # pdb.set_trace()
    
    df_classification = pd.DataFrame(classification_scores)
    # pdb.set_trace()
    df_classification.columns = ['precision', 'recall', 'fscore', 'support']
    df_classification.set_index(y_test.columns, inplace=True)

    # currently the child_alone column has labeled entries, thus dropping it for now
    df_classification.drop(['child_alone'], axis=0, inplace=True) 

    # below loop splits the precision, recall and f-score columns into two, one for negatives and one for positives (0 and 1)
    for column in df_classification.columns:
        column_1 = df_classification[column].apply(lambda x: x[0]).rename(column+str(0), inplace=True)
        # pdb.set_trace()
        column_2 = df_classification[column].apply(lambda x: x[1]).rename(column+str(1), inplace=True)
        # pdb.set_trace()
        df_classification.drop([column], axis=1, inplace=True)
        df_classification = pd.concat([df_classification, column_1, column_2], axis=1)
        # pdb.set_trace()

    # finally, take the average of the dataframe to get a classifier for the model                                                                    
    df_classification_avg = df_classification.mean(axis=0)
                                                                          
    return df_classification_avg

def evaluate_model(model, X_test, y_test): # took out the category names input here
    # as I think it makes more sense to iterate over the columns in y_test
    y_pred = model.predict(X_test)
    df_classification_avg_knn = classify_model_output(y_test, y_pred)
    return df_classification_avg_knn
    pass


def save_model(model, model_filepath):
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()