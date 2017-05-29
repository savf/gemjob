from dm_general import print_data_frame
import os
import json
import pandas as pd
import numpy as np
import random

_working_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
_percentage_few_missing = 0.01
_percentage_some_missing = 0.1
_percentage_too_many_missing = 0.5


def create_data_frame(file_name):
    """ Load data from json file and return as pandas DataFrame

    :param file_name: JSON filename
    :type file_name: str
    :return: DataFrame with data from JSON file
    :rtype: pandas.DataFrame
    """

    with open(_working_dir + file_name, "r") as f:
        found_jobs = f.read()
    data = json.loads(found_jobs)

    # normalize json because of nested client data
    df = pd.io.json.json_normalize(data)
    return df


def prepare_data(file_name):
    """ Clean data

    :param file_name: File name where data is stored
    :type file_name: str
    :return: Cleaned DataFrame
    :rtype: pandas.DataFrame
    """
    data_frame = create_data_frame(file_name)
    data_frame.columns = [c.replace('.', '_') for c in
                          data_frame.columns]  # so we can access a column with "data_frame.client_reviews_count"
    print_data_frame("Before changing data", data_frame)

    # set id
    data_frame.set_index("id", inplace=True)

    # remove unnecessary data
    unnecessary_columns = ["category2", "job_status", "url", "client_payment_verification_status"]
    data_frame.drop(labels=unnecessary_columns, axis=1, inplace=True)

    # convert total_charge and freelancer_count to number
    data_frame["total_charge"] = pd.to_numeric(data_frame["total_charge"])
    data_frame["freelancer_count"] = pd.to_numeric(data_frame["freelancer_count"])

    # handle missing values
    # ( data may change -> do this in a generic way! )

    # remove column if too many missing (removes duration)
    min_too_many_missing = missing_value_limit(data_frame.shape[0])
    columns_too_many_missing = list(data_frame.columns[data_frame.isnull().sum() > min_too_many_missing])
    data_frame.drop(labels=columns_too_many_missing, axis=1, inplace=True)

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

    # fill missing workload values with random non-missing values
    filled_workloads = data_frame["workload"].dropna()
    data_frame["workload"] = data_frame.apply(
        lambda row: row["workload"] if row["workload"] is not None else random.choice(filled_workloads), axis=1)

    ### add additional attributes like text size (how long is the description?) or number of skills
    data_frame["snippet_length"] = data_frame["snippet"].str.split().str.len()
    data_frame["skills_number"] = data_frame["skills"].str.len()

    print_data_frame("After preparing data", data_frame)

    return data_frame


def balance_data_set(data_frame, label_name):
    """ Balance the data set for classification (ratio of classes 1:1)

       :param data_frame: Pandas DataFrame that contains the data
       :type data_frame: pd.DataFrame
       :param label_name: Target label that will be learned
       :type label_name: str
       :return: Pandas DataFrame (balanced)
       :rtype: pandas.DataFrame
       """
    value_counts = data_frame[label_name].value_counts()
    min_target_value_count = min(value_counts.values)
    print "Value counts:\n", \
        value_counts, "\nminimum:", min_target_value_count,"\n ###\n"

    samples = []
    for value_class in value_counts.index:
        samples.append(data_frame.ix[data_frame[label_name] == value_class].sample(n=min_target_value_count, replace=False, random_state=0))
    data_frame = pd.concat(samples)

    print "Value counts:\n", \
        data_frame[label_name].value_counts(), "\nminimum:", min_target_value_count, "\n ###\n"

    return data_frame


def separate_text(data_frame, label_name):
    """ Separate structured data from text

    :param data_frame: Pandas DataFrame that contains the data
    :type data_frame: pd.DataFrame
    :param label_name: Target label that will be learned
    :type label_name: str
    :return: Pandas DataFrames once with only numerical attributes and once only text attributes
    :rtype: pandas.DataFrame
    """
    text_data = data_frame[[label_name, "skills", "snippet", "title"]]
    data_frame.drop(labels=["skills", "snippet", "title"], axis=1, inplace=True)

    return data_frame, text_data



def convert_to_numeric(data_frame, label_name):
    """ Convert client_country, job_type, subcategory2 and workload to numeric

    :param data_frame: Pandas DataFrame that contains the data
    :type data_frame: pd.DataFrame
    :param label_name: Target label that will be learned
    :type label_name: str
    :return: Cleaned Pandas DataFrames once with only numerical attributes and once only text attributes
    :rtype: pandas.DataFrame
    """
    # transform nominals client_country, job_type and subcategory2 to numeric
    cols_to_transform = ['client_country', 'job_type', 'subcategory2']
    data_frame = pd.get_dummies(data_frame, columns=cols_to_transform)

    # workload: has less than 10, 10-30 and 30+ -> convert to 5, 15 and 30?
    data_frame.ix[data_frame.workload == "Less than 10 hrs/week", 'workload'] = 5
    data_frame.ix[data_frame.workload == "10-30 hrs/week", 'workload'] = 15
    data_frame.ix[data_frame.workload == "30+ hrs/week", 'workload'] = 30
    data_frame["workload"] = pd.to_numeric(data_frame["workload"])

    return separate_text(data_frame, label_name)


def convert_to_nominal(data_frame, label_name):
    """ Convert all attributes in the given data_frame to nominal

    :param data_frame: Pandas DataFrame containing all data
    :type data_frame: pd.DataFrame
    :param label_name: Target label that will be learned
    :type label_name: str
    :return: Cleaned Pandas DataFrame once with only nominal attributes and once only text attributes
    :rtype: pandas.DataFrame
    """
    # TODO
    return separate_text(data_frame, label_name)


def missing_value_limit(data_frame_size):
    """ Calculates the amount of missing values that is tolerable

    :param data_frame_size: Total size of data frame
    :type data_frame_size: int
    :return: Limit
    :rtype: int
    """

    return data_frame_size * _percentage_too_many_missing
