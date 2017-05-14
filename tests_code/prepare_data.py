import os
import json
import pandas as pd

def printDF(title, df):
    print "##############################\n    "+title+"    \n##############################\n"
    print "## Shape: ##"
    print data_frame.shape
    print "\n## Missing Values per Column: ##"
    print data_frame.isnull().sum()
    # print "\n## Show data: ##"
    # print data_frame[0:3]
    print "############################## \n\n"

def createDF(file_name):
    # load data from json file
    working_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
    with open(working_dir + file_name, "r") as f:
        found_jobs = f.read()
    data = json.loads(found_jobs)

    # normalize because of nested client data
    return pd.io.json.json_normalize(data) 


data_frame = createDF("found_jobs_4K.json")

printDF("Before changing data", data_frame)

### remove unnecessary data
unnecessary_columns = ["id", "category2", "job_status", "url", "client.payment_verification_status"]
data_frame.drop(labels=unnecessary_columns, axis=1, inplace=True)

### handle missing values 
# ( data may change -> do this in a generic way! )

# remove rows that have missing data in columns, which normally only have very few (if any) missing values
max_few_missing = 30
columns_few_missing = list(data_frame.columns[data_frame.isnull().sum() < max_few_missing])
data_frame.dropna(subset=columns_few_missing, how='any')


printDF("After changing data", data_frame)
