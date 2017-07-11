from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split

from dm_data_preparation import *
from dm_general import evaluate_regression, print_predictions_comparison


def prepare_data_feedback_model(data_frame, label_name):
    """ Clean data specific to the feedback model

    :param data_frame: Pandas DataFrame that holds the data
    :type data_frame: pandas.DataFrame
    :param label_name: Target label that will be predicted
    :type label_name: str
    :return: Cleaned Pandas DataFrames once with only numerical attributes and once only text attributes
    :rtype: pandas.DataFrame
    """

    # drop columns where we don't have user data or are unnecessary
    drop_unnecessary = ["client_past_hires"]
    data_frame.drop(labels=drop_unnecessary, axis=1, inplace=True)

    # convert everything to numeric
    data_frame = convert_to_numeric(data_frame, label_name)

    # print data_frame, "\n"
    print_data_frame("After preparing for rating model", data_frame)

    return separate_text(data_frame, label_name)


# TODO: try classification instead of regression
def feedback_model(file_name):
    """ Learn model for label 'feedback' and return it

    :param file_name: JSON file containing all data
    :type file_name: str
    """
    label_name = "client_feedback"

    #data_frame = prepare_data(file_name)
    data_frame = load_data_frame_from_db()
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

    print_predictions_comparison(df_test, predictions, label_name, 50)

    # print_correlations(data_frame, label_name)
