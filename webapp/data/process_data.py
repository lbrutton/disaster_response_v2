# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
import pdb

def load_data(messages_filepath, categories_filepath):
    '''
    Load the data from the messages and categories csvs and add them to a pandas dataframe
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id', how='inner')
    return df
    pass


def clean_data(df):
    '''
    Clean the data stored in df - 
    in particular splitting the 'categories' column into 36 separate columns
    '''
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1:])
        
        # convert column from string to numeric
        categories[column] = categories[column].apply(lambda x: 1 if int(x) > 0 else 0)
    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # remove the child alone column as it has 0 entries
    # df.drop(['child_alone'], axis=1, inplace=True)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    print(df.related.unique())
    return df
    pass


def save_data(df, database_filename):
    '''
    Save the data from the pandas dataframe to the sqlite database
    '''
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('Messages', engine, index=False, if_exists='replace')
    pass  


def main():
    '''
    Take 4 arguments and run the functions in this file in order to process and clean the data
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