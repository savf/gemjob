import dm_data_preparation
from dm_data_preparation import *
from dm_text_mining import doTextMining
import random

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
    min_too_many_missing = dm_data_preparation._percentage_too_many_missing * data_frame.shape[0]
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
    label_name = "budget"
    data_frame = prepareData(file_name)
    data_frame = prepareDataBudgetModel(data_frame)

    df_train, df_test = splitIntoTestTrainSet(data_frame, 0.8)

    doTextMining(df_train, df_test, label_name, regression=True, max_features=5000)

    # printCorr(data_frame, label_name)
    # printCorr(data_frame, "total_charge")
