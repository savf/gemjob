from dm_general import print_data_frame, block_printing, enable_printing
import os
import json
import pandas as pd
import numpy as np
from math import log, exp
import random
import rethinkdb as rdb
from rethinkdb import RqlDriverError, RqlRuntimeError
import datetime as dt
from parameters import *

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


def create_data_frame(file_name, jobs=None):
    """ Load data from json file and return as pandas DataFrame

    :param file_name: JSON filename
    :type file_name: str
    :param jobs: Jobs as list of dict, not from file
    :type jobs: list(dict())
    :return: DataFrame with data from JSON file
    :rtype: pandas.DataFrame
    """
    if jobs is None:
        with open(_working_dir + file_name, "r") as f:
            found_jobs = f.read()
        jobs = json.loads(found_jobs)

    # normalize json because of nested client data
    df = pd.io.json.json_normalize(jobs)
    return df


def db_setup(file_name, host='localhost', port='28015', connection=None):
    """ Create DB and table if they don't exist, then insert jobs

    The database_module needs to be running and the host variable
    should be configured to the local host. For standard Docker
    installations, 'localhost' should work.

    :param file_name: File name where data is stored
    :type file_name: str
    :param host: RethinkDB host
    :type host: str
    :param port: RethinkDB port
    :type port: str
    :param connection: RethinkDB connection
    :type connection: rethinkdb.net.ConnectionInstance
    """
    database = 'datasets'
    prepared_jobs_table = 'jobs_optimized'
    if connection is None:
        connection = rdb.connect(host, port)
    try:
        if not rdb.db_list().contains(database).run(connection):
            rdb.db_create(database).run(connection)
        if not rdb.db(database).table_list().contains(prepared_jobs_table).run(connection):
            rdb.db(database).table_create(prepared_jobs_table).run(connection)
            data_frame = prepare_data(file_name)
            data_frame.date_created = data_frame.date_created.apply(
                lambda time: time.to_pydatetime().replace(
                    tzinfo=rdb.make_timezone("+02:00"))
            )
            data_frame.feedback_for_client.fillna(-1, inplace=True)
            data_frame.feedback_for_freelancer.fillna(-1, inplace=True)
            data_frame['id'] = data_frame.index
            rdb.db(database).table(prepared_jobs_table).insert(
                data_frame.to_dict('records'), conflict="replace").run(connection)
    except RqlRuntimeError as e:
        print 'Database error: {}'.format(e)

    return connection


def load_data_frame_from_db(connection=None, host='localhost', port='28015'):
    """ Load a prepared data_frame directly from the RethinkDB

    :param connection: RethinkDB connection
    :type connection: rethinkdb.net.ConnectionInstance
    :param host: RethinkDB host
    :type host: str
    :param port: RethinkDB port
    :type port: str
    :return: Prepared DataFrame
    :rtype: pandas.DataFrame
    """
    try:
        if connection is None:
            connection = rdb.connect(host, port)

        if rdb.db(RDB_DB).table(RDB_JOB_OPTIMIZED_TABLE).is_empty().run(connection):
            db_setup(JOBS_FILE, connection=connection, host=host, port=port)

        jobs_cursor = rdb.db(RDB_DB).table(RDB_JOB_OPTIMIZED_TABLE).run(connection)
        jobs = list(jobs_cursor)
        data_frame = pd.DataFrame(jobs)
        data_frame.set_index('id', inplace=True)
        data_frame['date_created'] = data_frame['date_created'].apply(
            lambda timestamp: pd.Timestamp(timestamp.replace(tzinfo=None))
        )
        data_frame.ix[data_frame.feedback_for_client == -1, 'feedback_for_client'] = None
        data_frame.ix[data_frame.feedback_for_freelancer == -1, 'feedback_for_freelancer'] = None

        return data_frame
    except RqlDriverError:
        # RethinkDB not reachable -> fall back to json file
        data_frame = prepare_data(JOBS_FILE)
        return data_frame
    except RqlRuntimeError as e:
        return None


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_str(s):
    try:
        str(s)
        return True
    except ValueError:
        return False


def is_correct_date(date_text):
    try:
        dt.datetime.strptime(date_text, '%m-%d-%Y')
        return True
    except ValueError:
        return False


def make_attributes_safe(raw_job):
    """ Check attribute types of raw job and correct if wrong type

    If experience_level is not correct, 2 (=intermediate) is chosen
    If job_type is not correct, "hourly" is chosen
    If start_date is not correct, todays date is chosen
    If snippet is too long, it's truncated to 5000 chars
    If title is too long, it's truncated to 500 chars
    If visibility is not correct, "public" is chosen

    :param raw_job: Job as dict
    :type raw_job: dict
    """
    for key, value in raw_job.iteritems():
        if key == 'budget':
            if not is_float(value):
                if not is_int(value):
                    raw_job[key] = 0
        elif key == 'client_country':
            if not is_str(value):
                raw_job[key] = ""
        elif key == 'client_feedback':
            if not is_int(value):
                raw_job[key] = 0
        elif key == 'client_reviews_count':
            if not is_int(value):
                raw_job[key] = 0
        elif key == 'client_jobs_posted':
            if not is_int(value):
                raw_job[key] = 0
        elif key == 'client_past_hires':
            if not is_int(value):
                raw_job[key] = 0
        elif key == 'duration':
            if not is_float(value):
                if not is_int(value):
                    raw_job[key] = 0
        elif key == 'duration_weeks_median':
            if not is_float(value):
                if not is_int(value):
                    raw_job[key] = 0
        elif key == 'experience_level':
            if is_int(value):
                if not 1 <= int(value) <= 3:
                    raw_job[key] = 2
            else:
                raw_job[key] = 2
        elif key == 'freelancer_count':
            if not is_int(value):
                raw_job[key] = 0
        elif key == 'job_type':
            if is_str(value):
                if value not in ["hourly", "fixed-price"]:
                    raw_job[key] = "hourly"
            else:
                raw_job[key] = "hourly"
        elif key == 'skills':
            if not is_str(value):
                raw_job[key] = ""
        elif key == 'start_date':
            if is_str(value):
                if not is_correct_date(value):
                    raw_job[key] = dt.date.today().strftime('%m-%d-%Y')
            else:
                raw_job[key] = dt.date.today().strftime('%m-%d-%Y')
        elif key == 'subcategory2':
            if not is_str(value):
                raw_job[key] = ""
        elif key == 'snippet':
            if is_str(value):
                if not len(value) <= 5000:
                    raw_job[key] = value[:4997] + "..."
            else:
                raw_job[key] = ""
        elif key == 'title':
            if is_str(value):
                if not len(value) <= 500 or not \
                        all([len(word) <= 50 for word in value.split()]):
                    raw_job[key] = value[:497] + "..."
            else:
                raw_job[key] = ""
        elif key == 'visibility':
            if is_str(value):
                if value not in ["public", "private", "invite-only"]:
                    raw_job[key] = "public"
            else:
                raw_job[key] = "public"
        elif key == 'workload':
            if is_str(value):
                if value not in ["10-30 hrs/week", "Less than 10 hrs/week", "30+ hrs/week"]:
                    raw_job[key] = "30+ hrs/week"
            else:
                raw_job[key] = "30+ hrs/week"


def prepare_data(file_name, jobs=None):
    """ Clean data

    :param file_name: File name where data is stored
    :type file_name: str
    :param jobs: List of jobs in dicts
    :type jobs: list(dict())
    :return: Cleaned DataFrame
    :rtype: pandas.DataFrame
    """
    data_frame = create_data_frame(file_name, jobs)
    data_frame.columns = [c.replace('.', '_') for c in
                          data_frame.columns]  # so we can access a column with "data_frame.client_reviews_count"

    # set id
    data_frame.set_index("id", inplace=True)

    # convert total_charge and freelancer_count to number
    data_frame["total_charge"] = pd.to_numeric(data_frame["total_charge"])
    data_frame["freelancer_count"] = pd.to_numeric(data_frame["freelancer_count"])

    # TODO: client_payment_verification_status may be static as well (or practically static)
    # Remove static and key attributes
    unnecessary_columns = ["category2", "job_status", "url"]
    data_frame.drop(labels=unnecessary_columns, axis=1, inplace=True)

    # generate aggregate feedback for client and freelancer and fill missings
    data_frame = get_overall_job_reviews(data_frame, drop_detailed=True)
    data_frame.feedback_for_client.fillna(-1, inplace=True)
    data_frame.feedback_for_freelancer.fillna(-1, inplace=True)

    # set missing of client_payment_verification_status to unknown (as this is already an option anyway)
    data_frame["client_payment_verification_status"].fillna("UNKNOWN", inplace=True)

    # remove duration and duration_weeks_total since duration_weeks_median is enough
    data_frame.drop(labels=['duration', 'duration_weeks_total'],
                    axis=1, inplace=True)

    # replace all workloads for fixed jobs with 30+ hrs/week to align the
    # missing ones and the ones which already were 30+ hrs/week
    data_frame.loc[
        data_frame['job_type'] == 'Fixed', 'workload'] = "30+ hrs/week"
    data_frame.dropna(subset=['workload'], how='any', inplace=True)

    # declare budget as missing, if hourly job (because there, we have no budget field)
    data_frame.loc[data_frame.job_type == "Hourly", 'budget'] = None

    # Fill with 0 since only hourly jobs have no budget and all non-missing
    # budgets for hourly jobs are set to 0
    data_frame.budget.fillna(0, inplace=True)

    # remove days and time from date_created to not fit to daily fluctuation
    data_frame['date_created'] = pd.to_datetime(data_frame['date_created'])
    data_frame['date_created'] = data_frame['date_created'].apply(lambda dt: dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0, nanosecond=0))

    # fill missing experience levels with forward filling
    data_frame['experience_level'].fillna(method='ffill', inplace=True)

    # drop missing values for total_hours, freelancer_count, duration_weeks_median
    data_frame.dropna(subset=["total_hours", "freelancer_count",
                              "duration_weeks_median"],
                      how='any', inplace=True)

    # add additional attributes like text size (how long is the description?) or number of skills
    data_frame["snippet_length"] = data_frame["snippet"].str.split().str.len()
    data_frame["title_length"] = data_frame["title"].str.split().str.len()
    data_frame["skills_number"] = data_frame["skills"].str.len()

    # rename A/B testing subcategory
    data_frame.loc[data_frame.subcategory2 == "a_b_testing", 'subcategory2'] = "A/B Testing"
    # print data_frame["subcategory2"].value_counts()

    # print_data_frame("After preparing data", data_frame)
    # print data_frame[0:3]

    data_frame.ix[data_frame.feedback_for_client == -1, 'feedback_for_client'] = None
    data_frame.ix[data_frame.feedback_for_freelancer == -1, 'feedback_for_freelancer'] = None

    return data_frame


def prepare_single_job(json_data):
    """ Prepare a single job to be similar to the prepared data

    :param json_data: Job in JSON format
    :type json_data: dict
    :return: Cleaned DataFrame
    :rtype: pandas.DataFrame
    """

    make_attributes_safe(json_data)

    data_frame = pd.DataFrame(json_data, index=[0])

    unnecessary_columns = ["visibility", "start_date", "url"]
    for c in unnecessary_columns:
        if c in data_frame.columns:
            data_frame.drop(labels=[c], axis=1, inplace=True)

    if 'duration' in data_frame.columns:
        data_frame['duration'] = data_frame.duration.astype(int)
        data_frame.rename(columns={'duration': 'total_hours'}, inplace=True)

    # capitalize hourly and fixed
    if 'job_type' in data_frame.columns:
        data_frame.loc[data_frame.job_type == "hourly", 'job_type'] = "Hourly"
        data_frame.loc[data_frame.job_type == "fixed-price", 'job_type'] = "Fixed"

    # Create date
    data_frame['date_created'] = dt.datetime.today()
    data_frame['date_created'] = data_frame['date_created'].apply(
        lambda dt: dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0, nanosecond=0))

    # declare budget as missing, if hourly job (because there, we have no budget field)
    if 'job_type' in data_frame.columns and 'budget' in data_frame.columns:
        data_frame.loc[data_frame.job_type == "Hourly", 'budget'] = None

        # Fill with 0 since only hourly jobs have no budget and all non-missing
        # budgets for hourly jobs are set to 0
        data_frame.budget.fillna(0, inplace=True)

    # convert experience level to numeric
    data_frame["experience_level"] = pd.to_numeric(data_frame["experience_level"], errors='coerce')  # errors are filled with NaN
    # print "# After to numeric:", data_frame["experience_level"]

    # add additional attributes like text size (how long is the description?) or number of skills
    if 'snippet' in data_frame.columns:
        data_frame["snippet_length"] = data_frame["snippet"].str.split().str.len()
    if 'title' in data_frame.columns:
        data_frame["title_length"] = data_frame["title"].str.split().str.len()
    if 'skills' in data_frame.columns:
        data_frame["skills_number"] = data_frame["skills"].str.split().str.len()

    # convert numeric attributes to numeric
    for c in ["budget", "client_feedback", "client_jobs_posted", "client_past_hires", "client_reviews_count", "duration_weeks_median", "freelancer_count", "total_charge", "total_hours"]:
        if c in data_frame.columns:
            data_frame[c] = pd.to_numeric(data_frame[c], errors='coerce') # errors are filled with NaN

    # create missing columns and fill with defaults
    default_values = {
        "budget": None,
        "client_country": None,
        "client_feedback": None,
        "client_jobs_posted": None,
        "client_past_hires": None,
        "client_payment_verification_status": None,
        "client_reviews_count": None,
        "date_created": None,
        "duration_weeks_median": None,
        "experience_level": None,
        "freelancer_count": None,
        "job_type": None,
        "skills": None,
        "snippet": None,
        "subcategory2": None,
        "title": None,
        "total_charge": None,
        "total_hours": None,
        "workload": None,
        "feedback_for_client": None,
        "feedback_for_freelancer": None,
        "snippet_length": None,
        "skills_number": None}

    for key, value in default_values.iteritems():
        if key not in data_frame.columns:
            data_frame[key] = value

    # print_data_frame("After preparing data", data_frame)
    # print data_frame

    return data_frame


# def treat_outliers(df_train, df_test, label_name="", budget_name="total_charge", add_to_df=False):
#     """ Delete examples with heavy outliers
#
#     :param df_train: Data Frame containing train data
#     :type df_train: pandas.DataFrame
#     :param df_test: Data Frame containing test data
#     :type df_test: pandas.DataFrame
#     :param label_name: Target label that will be learned
#     :type label_name: str
#     :param budget_name: Use either "budget" or "total_charge"
#     :type file_name: str
#     :param add_to_df: add log scale as new attributes (True) or replace old attributes (False)
#     :type add_to_df: bool
#     """
#     df_train = treat_outliers_deletion(df_train, budget_name)
#     df_train = treat_outliers_log_scale(df_train, label_name, budget_name, add_to_df=add_to_df)
#     df_test = treat_outliers_log_scale(df_test, label_name, budget_name, add_to_df=add_to_df)
#     return df_train, df_test


def treat_outliers_deletion(data_frame, ignore_labels=[]):
    """ Delete examples with heavy outliers
    delete only in training set!!!!

    :param data_frame: Data Frame
    :type data_frame: pandas.DataFrame
    :param budget_name: Use either "budget" or "total_charge"
    :type file_name: str
    """
    # delete only in training set!!!!

    attributes = ['total_hours', 'duration_weeks_median', 'total_charge']

    if data_frame['budget'].dtype.name != "category":
        attributes.append('budget')

    for attr in attributes:
        if attr in ignore_labels or attr not in data_frame.columns:
            attributes.remove(attr)

    q1 = data_frame[attributes].quantile(0.25)
    q3 = data_frame[attributes].quantile(0.75)
    iqr = q3 - q1

    outliers = ((data_frame[attributes] < (q1 - 1.5 * iqr)) | (data_frame[attributes] > (q3 + 1.5 * iqr)))

    outlier_indices = [idx for idx in data_frame[attributes].index if outliers[attributes].loc[idx].any()]
    del outliers
    data_frame.drop(outlier_indices, inplace=True)

    return data_frame


def transform_log_scale(data_frame, ignore_labels=[], add_to_df=False):
    """ Transform attributes with a lot of outliers/strong differences to log scale

    :param data_frame: Data Frame
    :type data_frame: pandas.DataFrame
    :param label_name: Label
    :type label_name: str
    :param add_to_df: add as new attributes (True) or replace old attributes (False)
    :type add_to_df: bool
    """
    attributes = ["total_hours",
                  "duration_weeks_total",
                  "duration_weeks_median",
                  "client_jobs_posted",
                  "client_reviews_count",
                  "client_past_hires",
                  'total_charge',
                  'feedback_for_client',
                  'feedback_for_freelancer',
                  'client_feedback']

    # no log for target label (budget or total_charge)

    if 'budget' in data_frame.columns and data_frame['budget'].dtype.name != "category":
            attributes.append('budget')

    prefix = ""
    if add_to_df:
        prefix = "log_"

    for attr in attributes:
        if attr in data_frame.columns and attr not in ignore_labels:
            data_frame[prefix+attr] = data_frame[attr].apply(lambda row: 0 if row < 1 else log(float(row)))

    return data_frame


def revert_log_scale(series):
    """ Revert log scale e.g. for predicted label

    :param series: Series
    :type series: pandas.Series
    """

    # Hint: There is no job with a budget of 1$ in the data (would not be allowed!) -> 0 is always 0
    series = series.apply(lambda row: 0 if row == 0 else exp(float(row)))

    return series


def balance_data_set(data_frame, label_name, relative_sampling=False, printing=True):
    """ Balance the data set for classification (ratio of classes 1:1)

    :param data_frame: Pandas DataFrame that contains the data
    :type data_frame: pd.DataFrame
    :param label_name: Target label that will be learned
    :type label_name: str
    :param relative_sampling: Relative or 1:1 sampling
    :type relative_sampling: bool
    :param printing: Whether to print info to console
    :type printing: bool
    :return: Pandas DataFrame (balanced)
    :rtype: pandas.DataFrame
    """
    if not printing:
        block_printing()

    print "### Balancing data:"
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

    enable_printing()
    return data_frame


def separate_text(data_frame, label_name=None):
    """ Separate structured data from text

    :param data_frame: Pandas DataFrame that contains the data
    :type data_frame: pd.DataFrame
    :param label_name: Target label that will be learned
    :type label_name: str
    :return: Pandas DataFrames once with only numerical attributes and once only text attributes
    :rtype: pandas.DataFrame
    """
    text_col_names = list(set(["skills", "snippet", "title"]).intersection(data_frame.columns))
    if label_name is None:
        text_data = data_frame[text_col_names]
    else:
        text_col_names.append(label_name)
        text_data = data_frame[text_col_names]
        text_col_names.remove(label_name)

    data_frame.drop(labels=text_col_names, axis=1, inplace=True)

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
    if 'date_created' in data_frame.columns:
        data_frame.rename(columns={'date_created': 'timestamp'}, inplace=True)
        data_frame['timestamp'] = pd.to_numeric(pd.to_timedelta(data_frame['timestamp']).dt.days)

    # transform nominals client_country, job_type and subcategory2 to numeric
    if label_name == 'job_type' or 'job_type' not in data_frame.columns:
        cols_to_transform = ['client_payment_verification_status', 'client_country', 'subcategory2']
    else:
        cols_to_transform = ['client_payment_verification_status', 'client_country', 'job_type', 'subcategory2']
    cols_to_transform = set(cols_to_transform).intersection(data_frame.columns)
    data_frame = pd.get_dummies(data_frame, columns=cols_to_transform)
    if 'workload' in data_frame.columns and not data_frame['workload'].empty:
        # workload: has less than 10, 10-30 and 30+ -> convert to 5, 15 and 30?
        data_frame.ix[data_frame.workload == "Less than 10 hrs/week", 'workload'] = 5
        data_frame.ix[data_frame.workload == "10-30 hrs/week", 'workload'] = 20
        data_frame.ix[data_frame.workload == "30+ hrs/week", 'workload'] = 30
        data_frame["workload"] = pd.to_numeric(data_frame["workload"])

    # print_data_frame("After converting to numeric", data_frame)
    # print data_frame[0:3]

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


def get_overall_job_reviews(data_frame, drop_detailed=True):
    """ Computes overall reviews from review categories

        :param data_frame: Pandas DataFrame with detailed feedbacks
        :type data_frame: pd.DataFrame
        :return: Pandas DataFrame with overall feedbacks
        :rtype: pandas.DataFrame
        """
    data_frame["feedback_for_client"] = data_frame[
        ['feedback_for_client_availability', 'feedback_for_client_communication',
         'feedback_for_client_cooperation', 'feedback_for_client_deadlines',
         'feedback_for_client_quality', 'feedback_for_client_skills']].mean(axis=1)
    data_frame["feedback_for_freelancer"] = data_frame[
        ['feedback_for_freelancer_availability', 'feedback_for_freelancer_communication',
         'feedback_for_freelancer_cooperation', 'feedback_for_freelancer_deadlines',
         'feedback_for_freelancer_quality', 'feedback_for_freelancer_skills']].mean(axis=1)

    if drop_detailed:
        data_frame.drop(labels=get_detailed_feedbacks_names(), axis=1, inplace=True)

    return data_frame


def normalize_z_score(data_frame, mean=None, std=None):
    """ Normalize based on mean and std

    :param data_frame: Pandas DataFrame
    :type data_frame: pd.DataFrame
    :param mean: mean values (optional)
    :type mean: pandas.Series
    :param std: std deviations (optional)
    :type std: pandas.Series
    :return: Normalized Pandas DataFrame
    :rtype: pandas.DataFrame
    """
    if mean is None or std is None:
        mean = data_frame.mean()
        std = data_frame.std()
    data_frame = (data_frame - mean) / std

    data_frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_frame.fillna(0, inplace=True)
    return data_frame, mean, std


def normalize_min_max(data_frame, min=None, max=None, classification_label=None):
    """ Normalize based on min and max values

    :param data_frame: Pandas DataFrame
    :type data_frame: pd.DataFrame
    :param min: minimum values (optional)
    :type min: pandas.Series
    :param max: maximum values (optional)
    :type max: pandas.Series
    :param classification_label: For classification, we need to exclude the label
    :type classification_label: str
    :return: Normalized Pandas DataFrame
    :rtype: pandas.DataFrame
    """
    if classification_label is not None:
        target_series = data_frame[classification_label]
        data_frame = data_frame.loc[:, data_frame.columns.difference(
            [classification_label])]

    if min is None or max is None:
        min = data_frame.min()
        max = data_frame.max()

    data_frame = (data_frame - min) / (max - min)

    if classification_label is not None:
        kwargs = {classification_label: target_series}
        data_frame = data_frame.assign(**kwargs)

    data_frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_frame.fillna(0, inplace=True)
    return data_frame, min, max


def denormalize_min_max(data_frame, min, max):
    if min is not None and max is not None:
        data_frame = data_frame * (max - min) + min

        data_frame.replace([np.inf, -np.inf], np.nan, inplace=True)
        data_frame.fillna(0, inplace=True)

    return data_frame


def weight_data(data_frame, text_weight=1.0):
    """ Weight certain attributes so they have less impact (countries and tokens)

    :param data_frame: Pandas DataFrame
    :type data_frame: pd.DataFrame
    :return: Weighted Pandas DataFrame
    :rtype: pandas.DataFrame
    """

    # data_frame["client_feedback"] = data_frame["client_feedback"] * 10
    # data_frame["duration_weeks_total"] = data_frame["duration_weeks_total"] * 10
    # data_frame["duration_weeks_median"] = data_frame["duration_weeks_median"] * 5
    # data_frame["freelancer_count"] = data_frame["freelancer_count"] * 10
    # data_frame["total_charge"] = data_frame["total_charge"] * 20
    # data_frame["skills_number"] = data_frame["skills_number"] * 5
    # data_frame["feedback_for_client"] = data_frame["feedback_for_client"] * 20
    # data_frame["feedback_for_freelancer"] = data_frame["feedback_for_freelancer"] * 10
    # data_frame["job_type_Fixed"] = data_frame["job_type_Fixed"] * 20
    # data_frame["job_type_Hourly"] = data_frame["job_type_Hourly"] * 20

    country_columns = [col for col in list(data_frame) if col.startswith('client_country')]
    data_frame[country_columns] = data_frame[country_columns] / len(country_columns)

    for text_column_name in ["skills", "snippet", "title"]:
        token_names = [col for col in list(data_frame) if col.startswith("$token_" + text_column_name)]
        if len(token_names) > 1:
            # print "Number of "+text_column_name+" tokens:", len(token_names)
            data_frame[token_names] = data_frame[token_names] * text_weight / len(token_names)

    data_frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_frame.fillna(0, inplace=True)
    return data_frame


def normalize_test_train(df_train, df_test, label_name=None, z_score_norm=False, weighting=True):
    """ Normalize and optionally weight train and test set (test set normalized based on train set!)

    :param df_train: Pandas DataFrame (train set)
    :type df_train: pd.DataFrame
    :param df_test: Pandas DataFrame (test set)
    :type df_test: pd.DataFrame
    :param z_score_norm: Use z-score normalization
    :type z_score_norm: bool
    :param weighting: Do weighting
    :type weighting: bool
    :return: Normalized Pandas DataFrames
    :rtype: pandas.DataFrame
    """

    if label_name is not None:
        # separate target
        df_target_train = df_train[label_name]
        df_train.drop(labels=[label_name], axis=1, inplace=True)
        df_target_test = df_test[label_name]
        df_test.drop(labels=[label_name], axis=1, inplace=True)

    if z_score_norm:
        df_train, mean, std = normalize_z_score(df_train)
        df_test, _, _ = normalize_z_score(df_test, mean, std)
    else:
        df_train, min, max = normalize_min_max(df_train)
        df_test, _, _ = normalize_min_max(df_test, min, max)

    if weighting:
        df_train = weight_data(df_train)
        df_test = weight_data(df_test)

    if label_name is not None:
        df_train = pd.concat([df_train, df_target_train], axis=1)
        df_test = pd.concat([df_test, df_target_test], axis=1)

    return df_train, df_test

def reduce_tokens_to_single_job(normalized_job, normalized_data, text_weight=5):
    """ Remove tokens not existing in a single job and reweight -> more focus on user text
    IMPORTANT: usually we should pass a copy of the data, so centroids are still the same for other jobs

    :param normalized_job: Pandas DataFrame
    :type normalized_job: pd.DataFrame
    :param normalized_data: Pandas DataFrame
    :type normalized_data: pd.DataFrame
    :return: Reweighted Pandas DataFrames with less tokens
    :rtype: pandas.DataFrame
    """
    # print "\n### reduce_tokens_to_single_job:"
    for text_column_name in ["skills", "snippet", "title"]:
        token_names = [col for col in list(normalized_job) if col.startswith("$token_" + text_column_name)]
        if len(token_names) > 1:
            # remove old weighting
            old_len = len(token_names)
            normalized_job[token_names] = normalized_job[token_names] * old_len
            normalized_data[token_names] = normalized_data[token_names] * old_len
            # remove tokens that are 0 in user job
            zero_tokens = list(normalized_job[token_names].loc[:, (normalized_job[token_names] == 0).any(axis=0)].columns)
            # print "\n## ZERO TOKENS: ##\n",zero_tokens[0:20]
            # print "# Removing", len(zero_tokens), "tokens and reweighting"
            normalized_job.drop(labels=zero_tokens, axis=1, inplace=True)
            normalized_data.drop(labels=zero_tokens, axis=1, inplace=True)

            # add new weighting
            remaining_columns = [x for x in token_names if x not in zero_tokens]
            new_len = old_len-len(zero_tokens)
            normalized_job[remaining_columns] = normalized_job[remaining_columns] * text_weight / new_len
            normalized_data[remaining_columns] = normalized_data[remaining_columns] * text_weight / new_len


    return normalized_job, normalized_data
