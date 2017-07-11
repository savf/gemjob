from dm_general import print_data_frame
import os
import json
import pandas as pd
import numpy as np
from math import log
import random
import rethinkdb as rdb
from rethinkdb import RqlDriverError, RqlRuntimeError
import datetime as dt

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


def db_setup(file_name):
    """ Create DB and table if they don't exist, then insert jobs

    The database_module needs to be running and the host variable
    should be configured to the local host. For standard Docker
    installations, 'localhost' should work.

    :param file_name: File name where data is stored
    :type file_name: str
    """
    host = '192.168.1.241'
    port = '28015'
    database = 'datasets'
    prepared_jobs_table = 'jobs_optimized'

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
        data_frame['id'] = data_frame.index
        rdb.db(database).table(prepared_jobs_table).insert(
            data_frame.to_dict('records'), conflict="replace").run(connection)
    except RqlRuntimeError:
        print 'Database {} and table {} already exist.'.format(database,
                                                               prepared_jobs_table)
    finally:
        connection.close()


def load_data_frame_from_db(connection=None):
    """ Load a prepared data_frame directly from the RethinkDB

    :return: Prepared DataFrame
    :rtype: pandas.DataFrame
    """
    host = '192.168.1.241'
    port = '28015'
    database = 'datasets'
    prepared_jobs_table = 'jobs_optimized'

    try:
        if connection is None:
            connection = rdb.connect(host, port)

        jobs_cursor = rdb.db(database).table(prepared_jobs_table).run(connection)
        jobs = list(jobs_cursor)
        data_frame = pd.DataFrame(jobs)
        data_frame.set_index('id', inplace=True)
        data_frame['date_created'] = data_frame['date_created'].apply(
            lambda timestamp: pd.Timestamp(timestamp.replace(tzinfo=None))
        )

        return data_frame
    except RqlDriverError:
        return None
    except RqlRuntimeError:
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
            if not is_int(value):
                raw_job[key] = -1
        elif key == 'client_country':
            if not is_str(value):
                raw_job[key] = ""
        elif key == 'client_feedback':
            if not is_int(value):
                raw_job[key] = 0
        elif key == 'client_reviews_count':
            if not is_int(value):
                raw_job[key] = 0
        elif key == 'duration':
            if not is_int(value):
                raw_job[key] = 0
        elif key == 'duration_weeks_median':
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

    # set id
    data_frame.set_index("id", inplace=True)

    # convert total_charge and freelancer_count to number
    data_frame["total_charge"] = pd.to_numeric(data_frame["total_charge"])
    data_frame["freelancer_count"] = pd.to_numeric(
        data_frame["freelancer_count"])

    # TODO: client_payment_verification_status may be static as well (or practically static)
    # Remove static and key attributes
    unnecessary_columns = ["category2", "job_status", "url"]
    data_frame.drop(labels=unnecessary_columns, axis=1, inplace=True)

    # generate aggregate feedback for client and freelancer and fill missings
    data_frame = get_overall_job_reviews(data_frame, drop_detailed=True)
    data_frame.feedback_for_client.fillna(method='ffill', inplace=True)
    data_frame.feedback_for_freelancer.fillna(method='ffill', inplace=True)

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

    # convert experience level from numeric to categorical
    experience_levels = ['Entry Level', 'Intermediate', 'Expert']
    data_frame['experience_level'] = pd.cut(data_frame['experience_level'], len(experience_levels),labels=experience_levels)

    # fill missing experience levels with forward filling
    data_frame['experience_level'].fillna(method='ffill', inplace=True)

    # drop missing values for total_hours, freelancer_count, duration_weeks_median
    data_frame.dropna(subset=["total_hours", "freelancer_count",
                              "duration_weeks_median"],
                      how='any', inplace=True)

    # add additional attributes like text size (how long is the description?) or number of skills
    data_frame["snippet_length"] = data_frame["snippet"].str.split().str.len()
    data_frame["skills_number"] = data_frame["skills"].str.len()

    # print_data_frame("After preparing data", data_frame)
    # print data_frame[0:3]

    return data_frame


def prepare_single_job(json_data):
    """ Prepare a single job to be similar to the prepared data

    :param json_data: Job in JSON format
    :type json_data: dict
    :return: Cleaned DataFrame
    :rtype: pandas.DataFrame
    """

    make_attributes_safe(json_data)
    # TODO check for correct types and values

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

    # convert experience level from numeric to categorical
    if 'experience_level' in data_frame.columns:
        data_frame['experience_level'] = data_frame.experience_level.astype(str)
        data_frame.loc[data_frame.experience_level == "1", 'experience_level'] = "Entry Level"
        data_frame.loc[data_frame.experience_level == "2", 'experience_level'] = "Intermediate"
        data_frame.loc[data_frame.experience_level == "3", 'experience_level'] = "Expert"

    # add additional attributes like text size (how long is the description?) or number of skills
    if 'snippet' in data_frame.columns:
        data_frame["snippet_length"] = data_frame["snippet"].str.split().str.len()
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


def treat_outliers(df_train, df_test, label_name="", budget_name="total_charge", add_to_df=False):
    """ Delete examples with heavy outliers

    :param df_train: Data Frame containing train data
    :type df_train: pandas.DataFrame
    :param df_test: Data Frame containing test data
    :type df_test: pandas.DataFrame
    :param label_name: Target label that will be learned
    :type label_name: str
    :param budget_name: Use either "budget" or "total_charge"
    :type file_name: str
    :param add_to_df: add log scale as new attributes (True) or replace old attributes (False)
    :type add_to_df: bool
    """
    df_train = treat_outliers_deletion(df_train, budget_name)
    df_train = treat_outliers_log_scale(df_train, label_name, budget_name, add_to_df=add_to_df)
    df_test = treat_outliers_log_scale(df_test, label_name, budget_name, add_to_df=add_to_df)
    return df_train, df_test


def treat_outliers_deletion(data_frame, budget_name="total_charge"):
    """ Delete examples with heavy outliers
    delete only in training set!!!!

    :param data_frame: Data Frame
    :type data_frame: pandas.DataFrame
    :param budget_name: Use either "budget" or "total_charge"
    :type file_name: str
    """
    # delete only in training set!!!!

    q1 = data_frame.quantile(0.25)
    q3 = data_frame.quantile(0.75)
    iqr = q3 - q1

    outliers = ((data_frame < (q1 - 1.5 * iqr)) | (data_frame > (q3 + 1.5 * iqr)))

    attributes_to_consider = ['total_hours', 'duration_weeks_median']

    if budget_name == "total_charge":
        attributes_to_consider.append('total_charge')
    elif data_frame['budget'].dtype.name != "category":
        attributes_to_consider.append('budget')

    # TODO: Not very efficient -> maybe something with apply and any?
    outlier_indices = [idx for idx in data_frame.index if
                       outliers[attributes_to_consider].loc[idx].any()]
    del outliers
    data_frame.drop(outlier_indices, inplace=True)

    return data_frame


def treat_outliers_log_scale(data_frame, label_name="", budget_name="total_charge", add_to_df=False):
    """ Transform attributes with a lot of outliers/strong differences to log scale

    :param data_frame: Data Frame
    :type data_frame: pandas.NDFrame
    :param label_name: Target label that will be learned
    :type label_name: str
    :param budget_name: Use either "budget" or "total_charge"
    :type file_name: str
    :param add_to_df: add as new attributes (True) or replace old attributes (False)
    :type add_to_df: bool
    """
    attributes = ["total_hours",
                  "duration_weeks_total",
                  "duration_weeks_median",
                  "client_jobs_posted",
                  "client_reviews_count",
                  "client_past_hires"]

    # no log for target label (budget or total_charge)

    if label_name != budget_name:
        if budget_name == "total_charge":
            attributes.append("total_charge")
        else:
            attributes.append("budget")

    prefix = ""
    if add_to_df:
        prefix = "log_"

    for attr in attributes:
        if attr in data_frame.columns:
            data_frame[prefix+attr] = data_frame[attr].apply(lambda row: 0 if row < 1 else log(float(row)))

    return data_frame


def balance_data_set(data_frame, label_name, relative_sampling=False):
    """ Balance the data set for classification (ratio of classes 1:1)

    :param data_frame: Pandas DataFrame that contains the data
    :type data_frame: pd.DataFrame
    :param label_name: Target label that will be learned
    :type label_name: str
    :param relative_sampling: Relative or 1:1 sampling
    :type relative_sampling: bool
    :return: Pandas DataFrame (balanced)
    :rtype: pandas.DataFrame
    """
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
        cols_to_transform = ['client_payment_verification_status', 'client_country', 'subcategory2', 'experience_level']
    elif label_name == 'experience_level':
        cols_to_transform = ['client_payment_verification_status', 'client_country', 'job_type', 'subcategory2']
    else:
        cols_to_transform = ['client_payment_verification_status', 'client_country', 'job_type', 'subcategory2', 'experience_level']
    cols_to_transform = set(cols_to_transform).intersection(data_frame.columns)
    data_frame = pd.get_dummies(data_frame, columns=cols_to_transform)

    if 'workload' in data_frame.columns:
        # workload: has less than 10, 10-30 and 30+ -> convert to 5, 15 and 30?
        data_frame.ix[data_frame.workload == "Less than 10 hrs/week", 'workload'] = 5
        data_frame.ix[data_frame.workload == "10-30 hrs/week", 'workload'] = 15
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


def normalize_min_max(data_frame, min=None, max=None):
    """ Normalize based on min and max values

    :param data_frame: Pandas DataFrame
    :type data_frame: pd.DataFrame
    :param min: minimum values (optional)
    :type min: pandas.Series
    :param max: maximum values (optional)
    :type max: pandas.Series
    :return: Normalized Pandas DataFrame
    :rtype: pandas.DataFrame
    """
    if min is None or max is None:
        min = data_frame.min()
        max = data_frame.max()
    data_frame = (data_frame - min) / (max - min)

    data_frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_frame.fillna(0, inplace=True)
    return data_frame, min, max


def weight_data(data_frame):
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

    token_names = [col for col in list(data_frame) if col.startswith('$token_')]
    if len(token_names) > 1:
        print "Number of text tokens:", len(token_names)
        # 3 text attributes, should roughly count as three
        data_frame[token_names] = data_frame[token_names] * 3 / len(token_names)

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
