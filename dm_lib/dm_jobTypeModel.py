from sklearn.ensemble import BaggingRegressor, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, \
    mutual_info_regression
from sklearn.model_selection import train_test_split

from dm_data_preparation import *
from dm_general import evaluate_classification, print_predictions_comparison
from dm_text_mining import add_text_tokens_to_data_frame


def prepare_data_jobtype_model(data_frame, label_name, relative_sampling):
    """ Prepare the given data to be used to predict the job type

    :param data_frame: Pandas DataFrame containing the data to be prepared
    :type data_frame: pandas.DataFrame
    :param label_name: Target label that will be predicted
    :type label_name: str
    :param relative_sampling: Relative or 1:1 sampling
    :type relative_sampling: Boolean
    :return: Cleaned Pandas DataFrames once with only nominal attributes and once only text attributes
    :rtype: pandas.DataFrame
    """

    # drop columns where we don't have user data or are unnecessary
    drop_unnecessary = ["client_payment_verification_status",
                        "feedback_for_client", "feedback_for_freelancer"]
    data_frame.drop(labels=drop_unnecessary, axis=1, inplace=True)

    # balance data set so ratio of hourly and fixed is 1:1
    data_frame = balance_data_set(data_frame, label_name,
                                  relative_sampling=relative_sampling,
                                  printing=False)

    # TODO convert everything to numeric? need that for quite a lot of classifiers
    data_frame = convert_to_numeric(data_frame, label_name)
    ### roughly cluster by rounding
    # data_frame = coarse_clustering(data_frame, label_name)

    # print data_frame, "\n"
    #print_data_frame("After preparing for job type model", data_frame)

    return data_frame


def prepare_single_job_jobtype_model(data_frame, label_name,
                                     columns, min, max, vectorizers):
    """ Prepare a data frame with a single job for prediction

    :param data_frame: Pandas DataFrame holding the single job
    :type data_frame: pandas.DataFrame
    :param label_name: Job type label to be predicted
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
    data_frame = prepare_data_jobtype_model(data_frame, label_name,
                                            relative_sampling=False)

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
    """ Create job type model for regression or classification

    :param df_train: Pandas DataFrame holding the data to be trained
    :type df_train: pandas.DataFrame
    :param label_name: job type label to be learned
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


def jobtype_model_development(file_name):
    """ Learn model for label 'job_type' and return it

    :param file_name: File name of JSON file containing the data
    :type file_name: str
    """
    label_name = "job_type"

    #data_frame = prepare_data(file_name)
    data_frame = load_data_frame_from_db()
    data_frame = prepare_data_jobtype_model(data_frame, label_name,
                                            relative_sampling=False)

    # print "\n\n########## Do Text Mining\n"
    # text_train, text_test = train_test_split(text_data, train_size=0.8)
    # do_text_mining(text_train, text_test, label_name, regression=False, max_features=5000)

    print "\n\n########## Classification based on all data including text\n"
    df_train, df_test = train_test_split(data_frame, train_size=0.8)

    df_train, df_train_text = separate_text(df_train, label_name)
    df_test, df_test_text = separate_text(df_test, label_name)

    df_train, vectorizers = add_text_tokens_to_data_frame(df_train,
                                                          df_train_text)
    df_test, _ = add_text_tokens_to_data_frame(df_test, df_test_text,
                                               vectorizers=vectorizers)

    model, columns = create_model(df_train, label_name,
                                  is_classification=True, selectbest=10)

    predictions = model.predict(df_test.ix[:, df_test.columns != label_name])

    evaluate_classification(df_test, predictions, label_name, printing=True)

    print_predictions_comparison(df_test, predictions, label_name)

    #with open("job_type_tree.dot", 'w') as f:
    #    f = tree.export_graphviz(clf, feature_names=df_train.columns.values, out_file=f)

    # print_correlations(data_frame, label_name)


def predict(data_frame, label_name, model, min=None, max=None):
    """ Predict job type for the given data frame

    :param data_frame: Pandas DataFrame holding the data for prediction
    :param label_name: Job type label
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


def jobtype_model_production(connection, normalization=True):
    """ Learn model for label 'job_type' on whole dataset and return it

    :param connection: RethinkDB connection
    :type connection: rethinkdb.net.ConnectionInstance
    :param normalization: Whether to do min-max normalization
    :type normalization: bool
    """
    label_name = 'job_type'
    jobtype_classification = True

    data_frame = load_data_frame_from_db(connection)
    data_frame = prepare_data_jobtype_model(data_frame, label_name,
                                            relative_sampling=False)

    data_frame = treat_outliers_deletion(data_frame)
    data_frame, text_data = separate_text(data_frame, label_name=label_name)
    data_frame, vectorizers = add_text_tokens_to_data_frame(data_frame,
                                                            text_data)
    if normalization:
        data_frame, min, max = normalize_min_max(data_frame,
                                                 classification_label=label_name)
    else:
        min, max = [None, None]

    model, columns = create_model(data_frame, label_name,
                                  jobtype_classification, selectbest=False)

    return model, columns, min, max, vectorizers
