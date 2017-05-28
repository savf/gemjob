import dm_data_preparation
from dm_data_preparation import *
from dm_general import evaluateRegression
from dm_text_mining import doTextMining
from sklearn import linear_model
from sklearn import svm
import random

def prepareDataBudgetModel(data_frame, label_name):
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
    min_too_many_missing = dm_data_preparation._percentage_too_many_missing * data_frame.shape[0]
    columns_too_many_missing = list(data_frame.columns[data_frame.isnull().sum() > min_too_many_missing])
    data_frame.drop(labels=columns_too_many_missing, axis=1, inplace=True)

    ### fill missing workload values with random non-missing values
    data_frame["workload"].fillna(random.choice(data_frame["workload"].dropna()), inplace=True)

    ### convert everything to numeric
    data_frame, text_data = convertToNumeric(data_frame, label_name)

    # print data_frame, "\n"
    printDF("After preparing for budget model", data_frame)

    return data_frame, text_data

def budgetModel(file_name):
    label_name = "budget"
    # label_name = "total_charge"

    data_frame = prepareData(file_name)
    # TODO step below removes text from data
    # -> do text mining right before that but after dropping rows without missing values
    # -> do it in "prepareDataBudgetModel()"
    data_frame, text_data = prepareDataBudgetModel(data_frame, label_name)

    print "\n\n########## Do Text Mining\n"
    text_train, text_test = splitIntoTestTrainSet(text_data, 0.8)
    doTextMining(text_train, text_test, label_name, regression=True, max_features=5000)

    print "\n\n########## Regression based on all data (except text)\n"
    df_train, df_test = splitIntoTestTrainSet(data_frame, 0.8)

    regr = svm.SVR(kernel='linear') #linear_model.Ridge(alpha=.5) #linear_model.LinearRegression()
    regr.fit(df_train.ix[:, df_train.columns != label_name], df_train[label_name])
    predictions = regr.predict(df_test.ix[:, df_train.columns != label_name])

    evaluateRegression(df_test, predictions, label_name)

    print "### Predictions: ###"
    print predictions[0:8]
    print "### Actual values: ###"
    print df_test[label_name][0:8]
    print "###########"

    # printCorr(data_frame, label_name)
