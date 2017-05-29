from dm_data_preparation import *
from dm_general import evaluate_regression, print_correlations
from dm_text_mining import do_text_mining
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import svm
from sklearn.preprocessing import LabelEncoder


def prepare_data_budget_model(data_frame, label_name):
    """ Clean data specific to the budget model

    :param data_frame: Pandas DataFrame that holds the data
    :type data_frame: pandas.DataFrame
    :param label_name: Target label that will be predicted
    :type label_name: str
    :return: Cleaned Pandas DataFrames once with only numerical attributes and once only text attributes
    :rtype: pandas.DataFrame
    """
    # remove rows with missing values

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

    # drop columns where we don't have user data or are unnecessary for budget
    drop_unnecessary = ["client_feedback", "client_reviews_count", "client_past_hires", "client_jobs_posted"]
    data_frame.drop(labels=drop_unnecessary, axis=1, inplace=True)

    # convert everything to numeric
    data_frame, text_data = convert_to_numeric(data_frame, label_name)

    # print data_frame, "\n"
    print_data_frame("After preparing for budget model", data_frame)

    return data_frame, text_data


# TODO: try classification instead of regression. Predict low budget (0 to x$), medium budget, ...
def budget_model(file_name):
    """ Learn model for label 'budget' and return it

    :param file_name: JSON file containing all data
    :type file_name: str
    """
    label_name = "budget"
    # label_name = "total_charge"

    data_frame = prepare_data(file_name)
    data_frame, text_data = prepare_data_budget_model(data_frame, label_name)

    print "\n\n########## Do Text Mining\n"
    text_train, text_test = train_test_split(text_data, train_size=0.8)
    do_text_mining(text_train, text_test, label_name, regression=True, max_features=5000)

    print "\n\n########## Regression based on all data (except text)\n"
    df_train, df_test = train_test_split(data_frame, train_size=0.8)

    regr = svm.SVR(kernel='linear')  # linear_model.Ridge(alpha=.5) #linear_model.LinearRegression()
    regr.fit(df_train.ix[:, df_train.columns != label_name], df_train[label_name])
    predictions = regr.predict(df_test.ix[:, df_train.columns != label_name])

    evaluate_regression(df_test, predictions, label_name)

    print "### Predictions: ###"
    print predictions[0:8]
    print "### Actual values: ###"
    print df_test[label_name][0:8]
    print "###########"

    # print_correlations(data_frame, label_name)
