import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load disaster data from the CSV file into a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    DataFrame: The loaded DataFrame.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    
    #print the shape of the data
    print("The merged file is of this size", df.shape)
    
    return df

def clean_data(df):
    """
    Clean the merged file by splitting the categories column into separate columns.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the column to be split.
    
    column_name : str
        The name of the column to be split by semicolon.
    
    Returns:
    -------
    pandas.DataFrame
        A DataFrame with the specified column split into separate rows.
    """
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    print('The new column names are:',category_colnames)
    categories.columns = category_colnames
    print("Split categories data:", categories.head(2))   
    
    # Remove the categories column from the main df 
    df.drop('categories', axis=1, inplace=True)
          
    # Replace the categories column with the new columns 
    df = pd.concat([df, categories], axis=1)
          
    # check for duplicates 
    duplicates = df.duplicated().sum()
    print(f"The data contains {duplicates} duplicates.")
    
    #drop the duplicates 
    df.drop_duplicates(inplace=True)
    print("The data is now this big",df.shape)
    
    # check if there are any remaining duplicates
    print('Duplicates remaining',df.duplicated().sum())
          
    return df

def save_data(df, database_filename):
    """
    save the cleaned file to sql db
    
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        
        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Disaster response data has been saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()