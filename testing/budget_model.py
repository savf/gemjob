import os
import json
import pandas as pd
# from pandas2arff import pandas2arff
import numpy as np
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

_working_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
_percentage_few_missing = 0.01
_percentage_some_missing = 0.1
_percentage_too_many_missing = 0.5

def printDF(title, df):
    print "##############################\n    "+title+"    \n##############################\n"
    print "## Shape: ##"
    print df.shape
    print "\n## Missing Values per Column: ##"
    print df.isnull().sum()
    print "\n## Data Types: ##"
    print df.dtypes
    # print "\n## Show data: ##"
    # print df[0:5]
    print "############################## \n\n"

def printCorr(df, attr=None):
    corr = df.corr()
    if attr==None:
        print "### Corrletaion Matrix ###"
        print corr
        plt.matshow(corr)
        plt.show()
    else:
        print "### Corrletaions for " + attr + " ###"
        print corr[attr].abs().sort_values(ascending=False)
    print "################################"

def createDF(file_name):
    # load data from json file
    
    with open(_working_dir + file_name, "r") as f:
        found_jobs = f.read()
    data = json.loads(found_jobs)

    # normalize json because of nested client data
    df = pd.io.json.json_normalize(data)    
    return df

def prepareData(file_name):
    data_frame = createDF(file_name)
    data_frame.columns = [c.replace('.', '_') for c in data_frame.columns] # so we can access a column with "data_frame.client_reviews_count"
    printDF("Before changing data", data_frame)

    ### set id
    data_frame.set_index("id", inplace=True)

    ### remove unnecessary data
    unnecessary_columns = ["category2", "job_status", "url", "client_payment_verification_status"]
    data_frame.drop(labels=unnecessary_columns, axis=1, inplace=True)

    ### convert total_charge and freelancer_count to number
    data_frame["total_charge"] = pd.to_numeric(data_frame["total_charge"])
    data_frame["freelancer_count"] = pd.to_numeric(data_frame["freelancer_count"])

    ### handle missing values
    # ( data may change -> do this in a generic way! )

    # remove rows that have missing data in columns, which normally only have very few (if any) missing values
    max_few_missing = _percentage_few_missing*data_frame.shape[0]
    columns_few_missing = list(data_frame.columns[(data_frame.isnull().sum() < max_few_missing) & (data_frame.isnull().sum() > 0)])
    data_frame.dropna(subset=columns_few_missing, how='any', inplace=True)
    
    # declare feedback as missing, if no reviews
    data_frame.ix[data_frame.client_reviews_count == 0, 'client_feedback'] = None
    # declare budget as missing, if 0  
    # TODO: good idea? would be 588 missing, now it's 2049; imo a budget of 0 is not setting a budget
    # data_frame.ix[data_frame.budget == 0, 'budget'] = None
    
    # convert date_created to timestamp as this accounts for changes in economy and prices (important for budget)
    data_frame.rename(columns={'date_created': 'timestamp'}, inplace=True)
    data_frame['timestamp'] = pd.to_numeric(pd.to_timedelta(pd.to_datetime(data_frame['timestamp'])).dt.days)

    # fill missing numeric values with mean, if only some missing 
    max_some_missing = _percentage_some_missing*data_frame.shape[0]
    df_numeric = data_frame.select_dtypes(include=[np.number])
    columns_some_missing = list(df_numeric.columns[(df_numeric.isnull().sum() < max_some_missing) & (df_numeric.isnull().sum() > 0)])
    data_frame[columns_some_missing] = data_frame[columns_some_missing].fillna((data_frame[columns_some_missing].mean()))
    del df_numeric

    ### add additional attributes like text size (how long is the description?) or number of skills
    data_frame["snippet_length"] = data_frame["snippet"].str.split().str.len()
    data_frame["skills_number"] = data_frame["skills"].str.len()

    printDF("After preparing data", data_frame)

    # pandas2arff(data_frame, "jobs.arff", wekaname = "jobs", cleanstringdata=True, cleannan=True)

    return data_frame

def cleanText(df, text_column_name, text_is_list):

    if text_is_list:
        df[text_column_name] = df.apply(lambda row: ' '.join([w.lower() for w in row[text_column_name]]), axis=1)
    else:
        stop_words = set(stopwords.words("english"))
        ps = PorterStemmer()
        df[text_column_name] = df.apply(lambda row: ' '.join([ps.stem(w).lower() for w in word_tokenize(row[text_column_name]) if not w in stop_words]), axis=1)

    return df

def prepareTextTrain(df, text_column_name, max_features, text_is_list=False):
    df = cleanText(df, text_column_name, text_is_list)

    # bag of words
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,
                                 max_features=max_features)
    train_data_features = vectorizer.fit_transform(df[text_column_name])
    train_data_features = train_data_features.toarray()

    return vectorizer, train_data_features

def prepareTextTest(df, text_column_name, vectorizer, text_is_list=False):
    df = cleanText(df, text_column_name, text_is_list)

    # bag of words
    test_data_features = vectorizer.transform(df[text_column_name])
    test_data_features = test_data_features.toarray()

    return test_data_features


def textRegressionModel(df, label_name, train_data_features):
    #TODO
    model = []
    return model

def textClassificationModel(df, label_name, train_data_features, n_estimators=100):
    # random forest classifier
    forest = RandomForestClassifier(n_estimators=n_estimators)
    forest = forest.fit(train_data_features, df[label_name])
    return forest

def testTextMining():
    print "\n\n### Testing Text Mining ###\n"
    max_features = 5000
    text_column_name = 'sentences'
    df_train = pd.DataFrame({text_column_name: ['This is a very good site. I will recommend it to others.', 'Aweful page, I hate it',
                                                'good work! keep it up', 'Terrible site, seriously aweful'],
                       "sentiment": ["pos", "neg", "pos", "neg"]})

    df_test = pd.DataFrame({text_column_name: ['This page is soo good!', 'This is really really good',
                                               'Your layout is seriously aweful', 'The most terrible site on the interwebs'],
                             "sentiment": ["pos", "pos", "neg", "neg"]})

    vectorizer, train_data_features = prepareTextTrain(df_train, text_column_name, max_features)
    test_data_features = prepareTextTest(df_test, text_column_name, vectorizer)

    print "\n## DataFrame: ##\n", df_train
    print "\n## Shape of Features: ##\n", train_data_features.shape
    print "\n## Features: ##\n", train_data_features
    vocab = vectorizer.get_feature_names()
    print "\n## Vocab: ##\n", vocab

    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data_features, axis=0)
    # For each, print the vocabulary word and the number of times it
    # appears in the training set
    print "\n## Frequency of words: ##"
    for tag, count in zip(vocab, dist):
        print count, tag

    model = textClassificationModel(df_train,"sentiment", train_data_features)

    print "\n## Model: ##\n", model

    predictions = model.predict(test_data_features)

    print "\n## Predictions: ##\n", predictions
    print "\n## Actual Sentiment: ##\n", df_test["sentiment"].to_dict().values()

    print "##############################"

def convertToNumeric(data_frame):
    
    # transform nominals client_country, job_type and subcategory2 to numeric
    cols_to_transform = [ 'client_country', 'job_type', 'subcategory2' ]
    data_frame = pd.get_dummies(data_frame, columns=cols_to_transform)

    # workload: has less than 10, 10-30 and 30+ -> convert to 5, 15 and 30?
    data_frame.ix[data_frame.workload == "Less than 10 hrs/week", 'workload'] = 5
    data_frame.ix[data_frame.workload == "10-30 hrs/week", 'workload'] = 15
    data_frame.ix[data_frame.workload == "30+ hrs/week", 'workload'] = 30
    data_frame["workload"] = pd.to_numeric(data_frame["workload"])

    ### predictions based on text: skills, snippet, subcategory2(?), title
    # TODO
    # DOES THAT EVEN WORK? CANT REPRODUCE CLUSTERS WITH EVAL DATA/USER DATA
    # remove text data for now TODO: undo that
    drop_columns = ["skills", "snippet", "title"]
    data_frame.drop(labels=drop_columns, axis=1, inplace=True)

    return data_frame

def prepareDataBudgetModel(data_frame):

    ### remove rows with missing values

    # rows that don't contain budget
    data_frame.dropna(subset=["budget"], how='any', inplace=True)

    # TODO just remove feedbacks?
    data_frame.dropna(subset=['feedback_for_client_availability', 'feedback_for_client_communication',
                              'feedback_for_client_cooperation', 'feedback_for_client_deadlines',
                              'feedback_for_client_quality', 'feedback_for_client_skills',
                              'feedback_for_freelancer_availability', 'feedback_for_freelancer_communication',
                              'feedback_for_freelancer_cooperation', 'feedback_for_freelancer_deadlines',
                              'feedback_for_freelancer_quality', 'feedback_for_freelancer_skills'],
                      how='any', inplace=True)

    ### drop columns where we don't have user data or are unnecessary for budget
    drop_unnecessary = ["client_feedback", "client_reviews_count", "client_past_hires", "client_jobs_posted"]
    data_frame.drop(labels=drop_unnecessary, axis=1, inplace=True)

    ### remove column if too many missing (removes duration)
    min_too_many_missing = _percentage_too_many_missing*data_frame.shape[0]
    columns_too_many_missing = list(data_frame.columns[data_frame.isnull().sum() > min_too_many_missing])
    data_frame.drop(labels=columns_too_many_missing, axis=1, inplace=True)

    ### fill missing workload values with random non-missing values 
    data_frame["workload"].fillna(random.choice(data_frame["workload"].dropna()), inplace=True)
    
    ### convert everything to numeric
    data_frame = convertToNumeric(data_frame)

    # print data_frame, "\n"
    printDF("After preparing for budget model", data_frame)

    return data_frame


def budgetModel(file_name):
    data_frame = prepareData(file_name)
    print data_frame["total_charge"][0:10]
    data_frame = prepareDataBudgetModel(data_frame)

    # printCorr(data_frame, "budget")
    # printCorr(data_frame, "total_charge")

#run
budgetModel("found_jobs_4K_extended.json")
# testTextMining()


