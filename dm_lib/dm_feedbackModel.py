from dm_data_preparation import *
from dm_general import evaluate_regression, print_correlations, print_predictions_comparison
from dm_text_mining import do_text_mining
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor


def prepare_data_feedback_model(data_frame, label_name):
    """ Clean data specific to the feedback model

    :param data_frame: Pandas DataFrame that holds the data
    :type data_frame: pandas.DataFrame
    :param label_name: Target label that will be predicted
    :type label_name: str
    :return: Cleaned Pandas DataFrames once with only numerical attributes and once only text attributes
    :rtype: pandas.DataFrame
    """

    # remove rows with missing values

    # remove other feedbacks
    # TODO summarize feedback to client to one value and predict this one instead of overall feedback
    data_frame.drop(labels=get_detailed_feedbacks_names(), axis=1, inplace=True)

    # drop columns where we don't have user data or are unnecessary
    drop_unnecessary = ["client_past_hires"]
    data_frame.drop(labels=drop_unnecessary, axis=1, inplace=True)

    # convert everything to numeric
    data_frame, text_data = convert_to_numeric(data_frame, label_name)
    ### roughly cluster by rounding
    # data_frame = coarse_clustering(data_frame, label_name)

    # print data_frame, "\n"
    print_data_frame("After preparing for rating model", data_frame)

    return data_frame, text_data


# TODO: try classification instead of regression
def feedback_model(file_name):
    """ Learn model for label 'feedback' and return it

    :param file_name: JSON file containing all data
    :type file_name: str
    """
    label_name = "client_feedback"

    data_frame = prepare_data(file_name)
    data_frame, text_data = prepare_data_feedback_model(data_frame, label_name)

    # print "\n\n########## Do Text Mining\n"
    # text_train, text_test = train_test_split(text_data, train_size=0.8)
    # do_text_mining(text_train, text_test, label_name, regression=True, max_features=5000)

    print "\n\n########## Regression based on all data (except text)\n"
    df_train, df_test = train_test_split(data_frame, train_size=0.8)

    regr = BaggingRegressor()#svm.SVR(kernel='linear')  # linear_model.Ridge(alpha=.5) #linear_model.LinearRegression()
    regr.fit(df_train.ix[:, df_train.columns != label_name], df_train[label_name])
    predictions = regr.predict(df_test.ix[:, df_test.columns != label_name])

    evaluate_regression(df_test, predictions, label_name)

    print_predictions_comparison(df_test, predictions, label_name, len(df_test))

    # print_correlations(data_frame, label_name)
