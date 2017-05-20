import os
import json
import pandas as pd
from pandas2arff import pandas2arff

def printDF(title, df):
    print "##############################\n    "+title+"    \n##############################\n"
    print "## Shape: ##"
    print df.shape
    print "\n## Missing Values per Column: ##"
    print df.isnull().sum()
    # print "\n## Show data: ##"
    # print df[0:12]
    print "############################## \n\n"

def createDF(file_name):
    # load data from json file
    working_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
    with open(working_dir + file_name, "r") as f:
        found_jobs = f.read()
    data = json.loads(found_jobs)

    # normalize because of nested client data
    df = pd.io.json.json_normalize(data)
    df.columns = [c.replace('.', '_') for c in df.columns] # so we can access a column with "data_frame.client_reviews_count"
    return df


data_frame = createDF("found_jobs_4K.json")

printDF("Before changing data", data_frame)

# Remove columns which add no additional value to the dataset
unnecessary_columns = ['category2', 'job_status', 'url', 'client_payment_verification_status']
data_frame.drop(labels=unnecessary_columns, axis=1, inplace=True)
# Remove examples which contribute missing values to attributes with < 30 missing attributes
max_few_missing = 30
columns_few_missing = list(data_frame.columns[data_frame.isnull().sum() < max_few_missing])
data_frame = data_frame.dropna(subset=columns_few_missing, how='any')

# Set feedback to None on examples where the client has not yet received a review
data_frame.loc[data_frame.client_reviews_count == 0, 'client_feedback'] = None
# Set budget to None if it is 0 (jobs without a budget cannot exist)
data_frame.loc[data_frame.budget == 0, 'budget'] = None
# Drop examples with missing values for budget and client_feedback
data_frame = data_frame.dropna(subset=['budget'], how='all')
data_frame = data_frame.dropna(subset=['client_feedback'], how='all')
# Completely drop attributes duration and workload
data_frame.drop(labels=['duration', 'workload'], axis=1, inplace=True)
# Transform nominals client_country, job_type and subcategory2 to numeric
clientCountryEncoder = LabelEncoder()
data_frame['client_country'] = clientCountryEncoder.fit_transform(data_frame['client_country'].astype('str'))
jobTypeEncoder = LabelEncoder()
data_frame['job_type'] = jobTypeEncoder.fit_transform(data_frame['job_type'].astype('str'))
subcategory2Encoder = LabelEncoder()
data_frame['subcategory2'] = subcategory2Encoder.fit_transform(data_frame['subcategory2'].astype('str'))
# Turn date_created into an epoch-like timestamp in days
data_frame['date_created'] = pd.to_numeric(pd.to_timedelta(pd.to_datetime(
    data_frame['date_created'])).dt.days)
data_frame.rename(columns={'date_created': 'timestamp'}, inplace=True)

# Save data_frame as ARFF file to experiment with in RapidMiner
pandas2arff(data_frame, "jobs.arff", wekaname = "jobs", cleanstringdata=True, cleannan=True)

printDF("After changing data", data_frame)
