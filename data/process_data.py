import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load Messages and Categories data from csv files
    
    Parameters:
        messages_filepath (str): messages csv file location
        categories_filepath (str): categories csv file location
        
    Return:
        df: Dataframe combine from messages and categories dataset
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    '''
    Clean the data frame
    
    Parameter:
        df: dataframe need to be cleaned
        
    Return:
        df: cleaned dataframe
    '''
    categories = df['categories'].str.split(';', expand=True)
    row = categories.head(1)
    category_colnames = row.applymap(lambda x: x[:-2]).iloc[0,:]
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
    df = df.drop(['categories'], axis=1)
    df = pd.concat([df, categories], axis=1, join='inner')
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    '''
    Save dataframe to the sqlite database
    
    Parameters:
        df: dataframe need to be stored
        database_filename: file name of sqlite database
    '''
    engine = create_engine(f'sqlite:///{database_filename}.db')
    df.to_sql('DisasterResponse', engine, index=False)  


def main():
    '''
    Main function to trigger the etl process
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()