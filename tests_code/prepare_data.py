import os
import json
import pandas as pd

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

### remove unnecessary data
unnecessary_columns = ["id", "category2", "job_status", "url", "client_payment_verification_status", "date_created"]
data_frame.drop(labels=unnecessary_columns, axis=1, inplace=True)

### handle missing values 
# ( data may change -> do this in a generic way! )

# remove rows that have missing data in columns, which normally only have very few (if any) missing values
max_few_missing = 30
columns_few_missing = list(data_frame.columns[data_frame.isnull().sum() < max_few_missing])
data_frame.dropna(subset=columns_few_missing, how='any')

# declare feedback as missing, if no reviews
data_frame.ix[data_frame.client_reviews_count == 0, 'client_feedback'] = None
# declare budget as missing, if 0 -> good idea? would be 588 missing, now it's 2049; imo a budget of 0 is not setting a budget
data_frame.ix[data_frame.budget == 0, 'budget'] = None

printDF("After changing data", data_frame)

