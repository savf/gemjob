from dm_data_preparation import *
from dm_general import evaluate_regression, evaluate_classification, print_correlations, print_predictions_comparison
from dm_text_mining import do_text_mining, addTextTokensToDF
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def prepare_data_budget_model(data_frame, label_name, budget_classification=False):
    """ Clean data specific to the budget model

    :param data_frame: Pandas DataFrame that holds the data
    :type data_frame: pandas.DataFrame
    :param label_name: Target label that will be predicted
    :type label_name: str
    :param budget_classification: Whether to prepare the data for a budget classification
    :type budget_classification: bool
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
    elif label_name == "budget":
        data_frame.drop(labels=["job_type"], axis=1, inplace=True)

        if budget_classification:
            # Discretize budget into 4 bins
            budget_levels = ['low', 'medium', 'high', 'ultra']
            data_frame['budget'] = pd.qcut(data_frame.loc[data_frame['budget'] > 0.0, 'budget'], len(budget_levels),
                                           labels=budget_levels)
            data_frame.loc[data_frame['budget'] == 0.0, 'budget'] = 'low'

    # remove rows with missing values

    # TODO just remove feedbacks?
    data_frame.drop(labels=get_detailed_feedbacks_names(), axis=1, inplace=True)

    # drop columns where we don't have user data or are unnecessary for budget
    drop_unnecessary = ["client_feedback", "client_reviews_count", "client_past_hires", "client_jobs_posted"]
    data_frame.drop(labels=drop_unnecessary, axis=1, inplace=True)

    # fill missing experience levels with random non-missing values
    filled_experience_levels = data_frame["experience_level"].dropna()
    data_frame["experience_level"] = data_frame.apply(
        lambda row: row["experience_level"] if row["experience_level"] is not None
        else random.choice(filled_experience_levels), axis=1)

    # convert everything to numeric
    data_frame = convert_to_numeric(data_frame, label_name)
    ### roughly cluster by rounding
    # data_frame = coarse_clustering(data_frame, label_name)

    # print data_frame, "\n"
    print_data_frame("After preparing for budget model", data_frame)

    return data_frame


# TODO: try classification instead of regression. Predict low budget (0 to x$), medium budget, ...
def budget_model(file_name):
    """ Learn model for label 'budget' and return it

    :param file_name: JSON file containing all data
    :type file_name: str
    """
    label_name = "budget"
    # label_name = "total_charge"

    data_frame = prepare_data(file_name, budget_name=label_name)

    # prepare both for model
    data_frame = prepare_data_budget_model(data_frame, label_name, budget_classification=True)

    # split
    df_train, df_test = train_test_split(data_frame, train_size=0.8)
    df_train, text_train = separate_text(df_train, label_name)
    df_test, text_test = separate_text(df_test, label_name)

    # print "\n\n########## Do Text Mining\n"
    # do_text_mining(text_train, text_test, label_name, regression=True, max_features=5000)

    # separate target
    df_target_train = df_train[label_name]
    df_train.drop(labels=[label_name], axis=1, inplace=True)
    df_target_test = df_test[label_name]
    df_test.drop(labels=[label_name], axis=1, inplace=True)

    print "\n\n########## Classification based on all data\n"
    # add tokens to data frame
    # df_train, df_test = addTextTokensToDF(df_train, df_test, text_train, text_test)
    # print_data_frame("After adding text tokens [TRAIN]", df_train)
    # print_data_frame("After adding text tokens [TEST]", df_test)

    #regr = BaggingRegressor()#svm.SVR(kernel='linear')  # linear_model.Ridge(alpha=.5) #linear_model.LinearRegression()
    #regr.fit(df_train, df_target_train)
    #predictions = regr.predict(df_test)

    #evaluate_regression(df_target_test, predictions, label_name)

    clf = RandomForestClassifier(n_estimators=100)

    clf.fit(df_train, df_target_train)
    predictions = clf.predict(df_test)

    evaluate_classification(df_target_test, predictions, label_name)

    print_predictions_comparison(df_target_test, predictions, label_name, 20)

    # print_correlations(data_frame, label_name)
