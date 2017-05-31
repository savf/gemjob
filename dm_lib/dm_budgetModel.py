from dm_data_preparation import *
from dm_general import evaluate_regression, print_correlations, print_predictions_comparison
from dm_text_mining import do_text_mining
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor


def prepare_data_budget_model(data_frame, label_name):
    """ Clean data specific to the budget model

    :param data_frame: Pandas DataFrame that holds the data
    :type data_frame: pandas.DataFrame
    :param label_name: Target label that will be predicted
    :type label_name: str
    :return: Cleaned Pandas DataFrames once with only numerical attributes and once only text attributes
    :rtype: pandas.DataFrame
    """
    # if we use total charge as budget, 0 values make no sense
    # (the budget would not be 0, we just didn't find a freelancer here)
    if label_name == "total_charge":
        # declare total_charge as missing, if 0
        data_frame.ix[data_frame.total_charge == 0, 'total_charge'] = None

        # rows that don't contain total_charge
        data_frame.dropna(subset=["total_charge"], how='any', inplace=True)

    # remove rows with missing values

    # TODO just remove feedbacks?
    data_frame.drop(labels=get_detailed_feedbacks_names(), axis=1, inplace=True)

    # drop columns where we don't have user data or are unnecessary for budget
    drop_unnecessary = ["client_feedback", "client_reviews_count", "client_past_hires", "client_jobs_posted"]
    data_frame.drop(labels=drop_unnecessary, axis=1, inplace=True)

    # convert everything to numeric
    data_frame, text_data = convert_to_numeric(data_frame, label_name)
    ### roughly cluster by rounding
    # data_frame = coarse_clustering(data_frame, label_name)

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

    data_frame = prepare_data(file_name, budget_name=label_name)
    data_frame, text_data = prepare_data_budget_model(data_frame, label_name)

    # print "\n\n########## Do Text Mining\n"
    # text_train, text_test = train_test_split(text_data, train_size=0.8)
    # do_text_mining(text_train, text_test, label_name, regression=True, max_features=5000)

    print "\n\n########## Regression based on all data (except text)\n"
    df_train, df_test = train_test_split(data_frame, train_size=0.8)

    regr = BaggingRegressor()#svm.SVR(kernel='linear')  # linear_model.Ridge(alpha=.5) #linear_model.LinearRegression()
    regr.fit(df_train.ix[:, df_train.columns != label_name], df_train[label_name])
    predictions = regr.predict(df_test.ix[:, df_train.columns != label_name])

    evaluate_regression(df_test, predictions, label_name)

    print_predictions_comparison(df_test, predictions, label_name, 20)

    # print_correlations(data_frame, label_name)
