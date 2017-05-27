from dm_general import printDF
import os
import json
import pandas as pd
import numpy as np

_working_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
_percentage_few_missing = 0.01
_percentage_some_missing = 0.1
_percentage_too_many_missing = 0.5

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
    data_frame.columns = [c.replace('.', '_') for c in
                          data_frame.columns]  # so we can access a column with "data_frame.client_reviews_count"
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
    max_few_missing = _percentage_few_missing * data_frame.shape[0]
    columns_few_missing = list(
        data_frame.columns[(data_frame.isnull().sum() < max_few_missing) & (data_frame.isnull().sum() > 0)])
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
    max_some_missing = _percentage_some_missing * data_frame.shape[0]
    df_numeric = data_frame.select_dtypes(include=[np.number])
    columns_some_missing = list(
        df_numeric.columns[(df_numeric.isnull().sum() < max_some_missing) & (df_numeric.isnull().sum() > 0)])
    data_frame[columns_some_missing] = data_frame[columns_some_missing].fillna(
        (data_frame[columns_some_missing].mean()))
    del df_numeric

    ### add additional attributes like text size (how long is the description?) or number of skills
    data_frame["snippet_length"] = data_frame["snippet"].str.split().str.len()
    data_frame["skills_number"] = data_frame["skills"].str.len()

    printDF("After preparing data", data_frame)

    # pandas2arff(data_frame, "jobs.arff", wekaname = "jobs", cleanstringdata=True, cleannan=True)

    return data_frame


def convertToNumeric(data_frame):
    # transform nominals client_country, job_type and subcategory2 to numeric
    cols_to_transform = ['client_country', 'job_type', 'subcategory2']
    data_frame = pd.get_dummies(data_frame, columns=cols_to_transform)

    # workload: has less than 10, 10-30 and 30+ -> convert to 5, 15 and 30?
    data_frame.ix[data_frame.workload == "Less than 10 hrs/week", 'workload'] = 5
    data_frame.ix[data_frame.workload == "10-30 hrs/week", 'workload'] = 15
    data_frame.ix[data_frame.workload == "30+ hrs/week", 'workload'] = 30
    data_frame["workload"] = pd.to_numeric(data_frame["workload"])

    ### predictions based on text: skills, snippet, subcategory2(?), title
    # TODO
    # remove text data for now TODO: undo that
    # drop_columns = ["skills", "snippet", "title"]
    # data_frame.drop(labels=drop_columns, axis=1, inplace=True)

    return data_frame

def convertToNominal(data_frame):
	# TODO
	return data_frame
	
def splitIntoTestTrainSet(df, size_train_set):
    train_indices = np.random.rand(len(df)) < size_train_set
    train = df[train_indices]
    test = df[~train_indices]
    return train, test
