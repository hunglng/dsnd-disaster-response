import sys
import re
import pandas as pd
import pickle
import numpy as np
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,  GridSearchCV 
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

def load_data(database_filepath):
    '''
    Load data from sqlite database
    
    Parameter
        database_filepath: sqlite database file location
        
    Return
        X: Messages dataframe
        y: target dataframe
        category_names: features list of target
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterResponse',engine)
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    '''
    Normalize, Split to word and Lemmatizer the input content
    
    Parameters:
        text (str): text input
        
    Return:
        list of root word after lemmatized
    
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    lemmatize  = [WordNetLemmatizer().lemmatize(w) for w in words]
    return lemmatize


def build_model():
    '''
    Build Classification model
    
    Return
        cv: classification model
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'tfidf__use_idf': [True, False],
        'clf__estimator__n_estimators': [10, 20, 50]
    }

    cv = GridSearchCV(pipeline, param_grid = parameters, verbose = 3)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Print classification report for each feature
    
    Parameters:
        model: classification model
        X_test: test dataframe
        Y_test: test target
        category_name: features list
    
    '''
    y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print(category_names[i], ":")
        print(classification_report(Y_test.iloc[:,i], y_pred[:,i]))


def save_model(model, model_filepath):
    '''
    Save model to pickle file
    
    Parameters:
        model: classification model
        model_filepath: picker file location
    '''
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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