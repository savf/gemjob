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
def get_detailed_feedbacks_names():
    return ['feedback_for_client_availability', 'feedback_for_client_communication',
                 'feedback_for_client_cooperation', 'feedback_for_client_deadlines',
                 'feedback_for_client_quality', 'feedback_for_client_skills',
                 'feedback_for_freelancer_availability', 'feedback_for_freelancer_communication',
                 'feedback_for_freelancer_cooperation', 'feedback_for_freelancer_deadlines',
                 'feedback_for_freelancer_quality', 'feedback_for_freelancer_skills']

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


def prepare_data(file_name, budget_name="total_charge"):
    """ Clean data

    :param file_name: File name where data is stored
    :type file_name: str
    :param budget_name: Use either "budget" or "total_charge"
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

    #exlusively work with one budget attribute
    if budget_name == "budget":
        # declare budget as missing, if hourly job (because there, we have no budget field)
        data_frame.ix[data_frame.job_type == "Hourly", 'budget'] = None

        # drop rows that don't contain budget
        data_frame.dropna(subset=["budget"], how='any', inplace=True)

        data_frame.drop(labels=["total_charge"], axis=1, inplace=True)
    else:
        data_frame.drop(labels=["budget"], axis=1, inplace=True)

    # remove days and time from date_created to not fit to daily fluctuation
    data_frame['date_created'] = pd.to_datetime(data_frame['date_created'])
    data_frame['date_created'] = data_frame['date_created'].apply(lambda dt: dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0, nanosecond=0))

    # convert experience level from numeric to categorical
    experience_levels = ['beginner', 'intermediate', 'expert']
    data_frame['experience_level'] = pd.cut(data_frame['experience_level'], len(experience_levels),
                                            labels=experience_levels)

    # fill missing numeric values with mean, if only some missing
    max_some_missing = _percentage_some_missing * data_frame.shape[0]
    df_numeric = data_frame.select_dtypes(include=[np.number])
    columns_some_missing = list(
        df_numeric.columns[(df_numeric.isnull().sum() < max_some_missing) & (df_numeric.isnull().sum() > 0)])
    data_frame[columns_some_missing] = data_frame[columns_some_missing].fillna(
        (data_frame[columns_some_missing].mean()))
    del df_numeric

    # # fill missing workload values with random non-missing values
    # filled_workloads = data_frame["workload"].dropna()
    # data_frame["workload"] = data_frame.apply(
    #     lambda row: row["workload"] if row["workload"] is not None else random.choice(filled_workloads), axis=1)

    # replace missing workloads with "Less than 10 hrs/week" because they are only smaller fixed jobs
    data_frame["workload"].fillna("Less than 10 hrs/week", inplace=True)

    ### add additional attributes like text size (how long is the description?) or number of skills
    data_frame["snippet_length"] = data_frame["snippet"].str.split().str.len()
    data_frame["skills_number"] = data_frame["skills"].str.len()

    print_data_frame("After preparing data", data_frame)

    return data_frame


def balance_data_set(data_frame, label_name, relative_sampling=False):
    """ Balance the data set for classification (ratio of classes 1:1)

       :param data_frame: Pandas DataFrame that contains the data
       :type data_frame: pd.DataFrame
       :param label_name: Target label that will be learned
       :type label_name: str
       :param relative_sampling: Relative or 1:1 sampling
       :type relative_sampling: Boolean
       :return: Pandas DataFrame (balanced)
       :rtype: pandas.DataFrame
       """
    value_counts = data_frame[label_name].value_counts()
    min_target_value_count = min(value_counts.values)
    print "Value counts:\n", \
        value_counts, "\nminimum:", min_target_value_count,"\n ###\n"

    samples = []
    if relative_sampling and len(value_counts) == 2:
        total_value_count = value_counts.sum()
        fractions = data_frame[label_name].value_counts().apply(lambda row: float(row) / total_value_count)
        for value_class in value_counts.index:
            samples.append(data_frame.loc[data_frame[label_name] == value_class].sample(frac=float(1.0 - fractions[value_class]), replace=False, random_state=0))
    else:
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
    :return: Cleaned Pandas DataFrames with everything converted to numeric except text
    :rtype: pandas.DataFrame
    """
    # convert date_created to timestamp as this accounts for changes in economy and prices (important for budget)
    data_frame.rename(columns={'date_created': 'timestamp'}, inplace=True)
    data_frame['timestamp'] = pd.to_numeric(pd.to_timedelta(data_frame['timestamp']).dt.days)

    # transform nominals client_country, job_type and subcategory2 to numeric
    if label_name == 'job_type' or 'job_type' not in data_frame.columns:
        cols_to_transform = ['client_country', 'subcategory2', 'experience_level']
    elif label_name == 'experience_level':
        cols_to_transform = ['client_country', 'job_type', 'subcategory2']
    else:
        cols_to_transform = ['client_country', 'job_type', 'subcategory2', 'experience_level']
    cols_to_transform = set(cols_to_transform).intersection(data_frame.columns)
    data_frame = pd.get_dummies(data_frame, columns=cols_to_transform)

    # workload: has less than 10, 10-30 and 30+ -> convert to 5, 15 and 30?
    data_frame.ix[data_frame.workload == "Less than 10 hrs/week", 'workload'] = 5
    data_frame.ix[data_frame.workload == "10-30 hrs/week", 'workload'] = 15
    data_frame.ix[data_frame.workload == "30+ hrs/week", 'workload'] = 30
    data_frame["workload"] = pd.to_numeric(data_frame["workload"])

    return data_frame


def coarse_clustering(data_frame, label_name):
    """ Roughly cluster data by rounding to remove effects of small variations (fluctuation)

    :param data_frame: Pandas DataFrame containing all data
    :type data_frame: pd.DataFrame
    :param label_name: Target label that will be learned
    :type label_name: str
    :return: Cleaned Pandas DataFrame once with only nominal attributes and once only text attributes
    :rtype: pandas.DataFrame
    """

    # TODO does rounding help or does it make things worse?

    # IMPORTANT: no missing values allowed when rounding -> remove those before

    # - budget: categorize (e.g. low, medium, high, very high)
    if "budget" in data_frame.columns:
        data_frame["budget"] = data_frame["budget"].round(-1)

    # - feedback: round to 1 decimal
    if "client_feedback" in data_frame.columns:
        data_frame["client_feedback"] = data_frame["client_feedback"].round(1)

    # - other client_* attributes
    if "client_jobs_posted" in data_frame.columns:
        data_frame["client_jobs_posted"] = data_frame["client_jobs_posted"].round(-1)
    if "client_reviews_count" in data_frame.columns:
        data_frame["client_reviews_count"] = data_frame["client_reviews_count"].round(-1)

    # - feedback_for_*: round to 1 decimal
    for fn in get_detailed_feedbacks_names():
        if fn in data_frame.columns:
            data_frame[fn] = data_frame[fn].round(1)

    # - freelancer_count: no changes

    # - total_charge: categorize like budget
    if "total_charge" in data_frame.columns:
        data_frame["total_charge"] = data_frame["total_charge"].round(-1)

    # - snippet_length: round to 10s
    if "snippet_length" in data_frame.columns:
        data_frame["snippet_length"] = data_frame["snippet_length"].round(-1)

    # - skills_number: no changes

    # - timestamp: round to 100s
    if "timestamp" in data_frame.columns:
        data_frame["timestamp"] = data_frame["timestamp"].round(-10)

    return data_frame


def missing_value_limit(data_frame_size):
    """ Calculates the amount of missing values that is tolerable

    :param data_frame_size: Total size of data frame
    :type data_frame_size: int
    :return: Limit
    :rtype: int
    """

    return data_frame_size * _percentage_too_many_missing
