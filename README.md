# Disaster Response Pipeline Project

### Table of Content
1. [Libraries](#libraries)
2. [Project Summary](#summary)
3. [File Descriptions](#files)
4. [Instructions](#instructions)


## Libraries <a name="libraries"></a>
This project uses Python 3.* with following libraries
- pandas
- re
- sys
- json
- sklearn
- nltk
- sqlalchemy
- pickle
- Flask
- plotly
- sqlite3

To install nltk package

`python -m nltk.downloader wordnet punkt stopwords`

## Project Summary <a name="summary"></a>
This project aim to classificate the messages into different categories base on the dataset provided by Figure Eight.

With 2 provided datasets disaster_categories and disaster_messages, We will 
- Build an ETL pipeline to merge 2 datasets, clean data and store them to sqlite database
- Build a ML pipeline to load data from sqlite database, do text processing, use GridSearchCV to train the model then export the model to a pickle file
- Build a Flask Web app for user to classify the input message

## File Descriptions <a name="files"></a>
```
- | data
    -- disaster_categories.csv categories dataset
    -- disaster_messages.csv messages dataset
    -- process_data.py ETL pipeline
- | model
    -- train_classifier Machine Learning pipeline
- | app
    -- run.py flask app
    - | template
        -- master.html main page
        -- go.html result page
```

## Instructions <a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database

        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
    
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
