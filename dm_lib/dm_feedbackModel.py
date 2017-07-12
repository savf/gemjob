from sklearn.ensemble import BaggingRegressor, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, \
    mutual_info_regression
from sklearn.model_selection import train_test_split

from dm_data_preparation import *
from dm_general import evaluate_regression, print_predictions_comparison
from dm_text_mining import add_text_tokens_to_data_frame


def prepare_data_feedback_model(data_frame, label_name):
    """ Clean data specific to the feedback model

    :param data_frame: Pandas DataFrame that holds the data
    :type data_frame: pandas.DataFrame
    :param label_name: Target label that will be predicted
    :type label_name: str
    :return: Cleaned Pandas DataFrames
    :rtype: pandas.DataFrame
    """

    # drop columns where we don't have user data or are unnecessary
    drop_unnecessary = ["client_country", "client_jobs_posted",
                        "client_past_hires",
                        "client_payment_verification_status",
                        "feedback_for_client", "feedback_for_freelancer"]
    data_frame.drop(labels=drop_unnecessary, axis=1, inplace=True)

    # convert everything to numeric
    data_frame = convert_to_numeric(data_frame, label_name)

    # print data_frame, "\n"
    print_data_frame("After preparing for feedback model", data_frame)

    return data_frame


def prepare_single_job_feedback_model(data_frame, label_name,
                                      columns, min, max, vectorizers):
    """ Prepare a data frame with a single job for prediction

    :param data_frame: Pandas DataFrame holding the single job
    :type data_frame: pandas.DataFrame
    :param label_name: Feedback label to be predicted
    :type label_name: str
    :param columns: List of columns, which need to be present
    :type columns: list(str)
    :param min: Minimum for min-max normalization
    :param max: Maximum for min-max normalization
    :param vectorizers: Vectorizers used for the text columns
    :type vectorizers: list(CountVectorizer)
    :return: Data Frame with single job ready for prediction
    :rtype: pandas.DataFrame
    """
    data_frame = prepare_data_feedback_model(data_frame, label_name=label_name)

    # handle text
    data_frame, text_data = separate_text(data_frame)
    if vectorizers is not None:
        data_frame, _ = add_text_tokens_to_data_frame(data_frame, text_data,
                                                      vectorizers=vectorizers)

    # add missing columns (dummies, that were not in this data set)
    for col in columns:
        if col not in data_frame.columns:
            data_frame[col] = 0
    # remove columns not existing in clusters
    for col in data_frame.columns:
        if col not in columns:
            data_frame.drop(labels=[col], axis=1, inplace=True)

    # normalize
    if min is not None and max is not None:
        data_frame, _, _ = normalize_min_max(data_frame, min, max)

    # order according to cluster_columns, since scikit does not look at labels!
    data_frame = data_frame.reindex_axis(columns, axis=1)

    return data_frame


def create_model(df_train, label_name, is_classification, selectbest=False):
    """ Create feedback model for regression or classification

    :param df_train: Pandas DataFrame holding the data to be trained
    :type df_train: pandas.DataFrame
    :param label_name: Feedback label to be learned
    :type label_name: str
    :param is_classification: Whether classification should be used
    :type is_classification: bool
    :param selectbest: False (=0) or number of features to be used
    :type selectbest: int
    :return: Model and columns of dataset
    """
    # separate target
    df_target_train = df_train[label_name]
    df_train.drop(labels=[label_name], axis=1, inplace=True)

    if selectbest > 0:
        if is_classification:
            selector = SelectKBest(f_classif, k=selectbest)
        else:
            selector = SelectKBest(mutual_info_regression, k=selectbest)
        selector.fit(df_train, df_target_train)
        relevant_indices = selector.get_support(indices=True)
        df_train = df_train.iloc[:, relevant_indices]
    if not is_classification:
        model = BaggingRegressor()  # svm.SVR(kernel='linear')  # linear_model.Ridge(alpha=.5) #linear_model.LinearRegression()
    else:
        model = RandomForestClassifier(n_estimators=100)

    model.fit(df_train, df_target_train)
    return model, df_train.columns


# TODO: try classification instead of regression
def feedback_model_development(file_name):
    """ Learn model for label 'feedback' and return it

    :param file_name: JSON file containing all data
    :type file_name: str
    """
    label_name = "client_feedback"
    feedback_classification = False

    #data_frame = prepare_data(file_name)
    data_frame = load_data_frame_from_db()
    data_frame = prepare_data_feedback_model(data_frame, label_name)

    print "\n\n########## Regression including text\n"
    df_train, df_test = train_test_split(data_frame, train_size=0.8)

    df_train, df_train_text = separate_text(df_train, label_name)
    df_test, df_test_text = separate_text(df_test, label_name)

    df_train, vectorizers = add_text_tokens_to_data_frame(df_train,
                                                          df_train_text)
    df_test, _ = add_text_tokens_to_data_frame(df_test, df_test_text,
                                               vectorizers=vectorizers)

    model, _ = create_model(df_train, label_name,
                            feedback_classification, selectbest=10)

    predictions = model.predict(df_test.ix[:, df_test.columns != label_name])

    evaluate_regression(df_test, predictions, label_name)

    print_predictions_comparison(df_test, predictions, label_name, 50)

    # print_correlations(data_frame, label_name)


def predict(data_frame, label_name, model, min=None, max=None):
    """ Predict feedback for the given data frame

    :param data_frame: Pandas DataFrame holding the data for prediction
    :param label_name: Feedback label
    :param model: Prediction model
    :param min: Minimum for min-max denormalization
    :param max: Maximum for min-max denormalization
    :return: Budget prediction
    """
    prediction = model.predict(data_frame)
    if len(prediction) > 0:
        prediction = prediction[0]
    else:
        return -1

    # De-normalize if min and max given
    if min is not None and max is not None:
        prediction_frame = pd.DataFrame()
        prediction_frame.set_value(0, label_name, prediction)
        prediction_frame = denormalize_min_max(prediction_frame,
                                               min=min, max=max)
        if len(prediction_frame[label_name]) > 0:
            prediction = prediction_frame[label_name][0]

    return prediction


def feedback_model_production(connection, label_name='client_feedback',
                              normalization=True):
    """ Learn model for label 'client_feedback' on whole dataset and return it
    
    :param connection: RethinkDB connection
    :type connection: rethinkdb.net.ConnectionInstance
    :param label_name: Feedback label to predict (default 'client_feedback')
    :type label_name: str
    :param normalization: Whether to do min-max normalization
    :type normalization: bool
    """
    feedback_classification = False

    data_frame = load_data_frame_from_db(connection)
    data_frame = prepare_data_feedback_model(data_frame, label_name)

    data_frame = treat_outliers_deletion(data_frame)
    data_frame, text_data = separate_text(data_frame, label_name=label_name)
    data_frame, vectorizers = add_text_tokens_to_data_frame(data_frame,
                                                            text_data)
    if normalization:
        data_frame, min, max = normalize_min_max(data_frame)
    else:
        min, max = [None, None]

    model, columns = create_model(data_frame, label_name,
                                  feedback_classification, selectbest=False)

    return model, columns, min, max, vectorizers
