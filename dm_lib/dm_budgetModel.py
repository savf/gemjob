from dm_general import *
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, \
    mutual_info_regression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split

from dm_data_preparation import *
from dm_text_mining import add_text_tokens_to_data_frame


def prepare_data_budget_model(data_frame, label_name, budget_classification=False):
    """ Clean data specific to the budget model

    :param data_frame: Pandas DataFrame that holds the data
    :type data_frame: pandas.DataFrame
    :param label_name: Target label that will be predicted
    :type label_name: str
    :param budget_classification: Whether to prepare the data for a budget classification
    :type budget_classification: bool
    :return: Cleaned Pandas DataFrames with only numerical attributes and the three text attributes
    :rtype: pandas.DataFrame
    """

    # drop all examples with job_type hourly since there is no budget or
    # total charge present
    data_frame.loc[data_frame['job_type'] == 'Hourly', 'job_type'] = None
    data_frame.dropna(subset=['job_type'], how='any', inplace=True)
    # job type is static for budget and total_charge (=fixed) so drop it
    data_frame.drop(labels=["job_type"], axis=1, inplace=True)

    # exclusively work with one budget attribute
    if label_name == "budget":
        data_frame.drop(labels=["total_charge"], axis=1, inplace=True)
        data_frame.dropna(subset=["budget"], how='any', inplace=True)

        # Discretize budget into 4 bins for classification
        if budget_classification:
            budget_levels = ['low', 'medium', 'high', 'ultra']
            data_frame['budget'] = pd.qcut(
                data_frame.loc[data_frame['budget'] > 0.0, 'budget'],
                len(budget_levels),
                labels=budget_levels)
            data_frame.loc[data_frame['budget'] == 0.0, 'budget'] = 'low'
    elif label_name == "total_charge":
        data_frame.drop(labels=["budget"], axis=1, inplace=True)
        # declare total_charge as missing, if 0
        data_frame.loc[data_frame.total_charge == 0, 'total_charge'] = None
        data_frame.dropna(subset=['total_charge'], how='any', inplace=True)

    # drop columns where we don't have user data or are unnecessary for budget
    drop_unnecessary = ["client_country", "client_jobs_posted",
                        "client_past_hires", "client_payment_verification_status",
                        "feedback_for_client", "feedback_for_freelancer"]
    data_frame.drop(labels=drop_unnecessary, axis=1, inplace=True)

    # convert everything to numeric
    data_frame = convert_to_numeric(data_frame, label_name)

    # print data_frame, "\n"
    # print_data_frame("After preparing for budget model", data_frame)

    return data_frame


def prepare_single_job_budget_model(data_frame, label_name,
                                    columns, min, max, vectorizers):
    """ Prepare a data frame with a single job for prediction

    :param data_frame: Pandas DataFrame holding the single job
    :type data_frame: pandas.DataFrame
    :param label_name: Budget label to be predicted
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
    data_frame = prepare_data_budget_model(data_frame, label_name=label_name,
                                           budget_classification=False)

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
    """ Create budget model for regression or classification

    :param df_train: Pandas DataFrame holding the data to be trained
    :param label_name: Budget label to learn ('budget' or 'total_charge')
    :param is_classification: Whether classification should be used
    :param selectbest: False or number of features to select
    :return: Model and selected columns
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


def create_model_cross_val(data_frame, label_name, is_classification):
    # separate target
    data_frame_target = data_frame[label_name]
    data_frame.drop(labels=[label_name], axis=1, inplace=True)

    if not is_classification:
        model = BaggingRegressor()  # svm.SVR(kernel='linear')  # linear_model.Ridge(alpha=.5) #linear_model.LinearRegression()
    else:
        model = RandomForestClassifier(n_estimators=100)

    print "\n###Cross Validation: "

    scores = cross_val_score(model, data_frame, data_frame_target, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    predictions = cross_val_predict(model, data_frame, data_frame_target, cv=10)

    return model, predictions, data_frame_target


# TODO: try classification instead of regression. Predict low budget (0 to x$), medium budget, ...
def budget_model(file_name, connection):
    """ Learn model for label 'budget' and return it

    :param file_name: JSON file containing all data
    :type file_name: str
    :param connection: RethinkDB connection to load the data
    :type connection: rethinkdb.net.ConnectionInstance
    """
    label_name = "budget"
    budget_classification = False
    do_cross_val = False
    # label_name = "total_charge"

    # data_frame = prepare_data(file_name)
    data_frame = load_data_frame_from_db(connection=connection)

    # prepare for model
    data_frame = prepare_data_budget_model(data_frame, label_name, budget_classification=budget_classification)
    # print_correlations(data_frame, label_name)

    if not do_cross_val:
        # split
        df_train, df_test = train_test_split(data_frame, train_size=0.8)

        # treat outliers
        df_train_outl, df_test_outl = treat_outliers(df_train.copy(), df_test.copy(), label_name, label_name, add_to_df=True)

        # separate text
        df_train, text_train = separate_text(df_train)
        df_test, text_test = separate_text(df_test)
        # separate text after outlier treatment
        df_train_outl, text_train_outl = separate_text(df_train_outl)
        df_test_outl, text_test_outl = separate_text(df_test_outl)

        # print "\n\n########## Do Text Mining\n"
        # do_text_mining(text_train, text_test, label_name, regression=True, max_features=5000)

        # print "\n\n##### With Outlier Treatment:"
        # model, _ = create_model(df_train_outl.copy(), label_name, budget_classification)
        # print_model_evaluation(model, df_test_outl.copy(), label_name, budget_classification)

        print "##### Without Outlier Treatment:"
        model, _ = create_model(df_train.copy(), label_name, budget_classification, selectbest=True)
        print_model_evaluation(model, df_test.copy(), label_name, budget_classification)

        # print "##### With Text Tokens, With Outlier Treatment:"
        # # add tokens to data frame
        # df_train_outl, vectorizers = add_text_tokens_to_data_frame(df_train_outl, text_train_outl)
        # df_test_outl, _ = add_text_tokens_to_data_frame(df_test_outl, text_test_outl, vectorizers=vectorizers)
        # model, _ = create_model(df_train_outl.copy(), label_name, budget_classification)
        # print_model_evaluation(model, df_test_outl.copy(), label_name, budget_classification)
        #
        # print "##### With Text Tokens, With Outlier Treatment, With Normalization, With Weighting:"
        # df_train_outl, df_test_outl = normalize_test_train(df_train_outl, df_test_outl, label_name=label_name, z_score_norm=False, weighting=True)
        # model, _ = create_model(df_train_outl, label_name, budget_classification)
        # print_model_evaluation(model, df_test_outl, label_name, budget_classification)
        #
        # print "##### With Text Tokens, Without Outlier Treatment:"
        # # add tokens to data frame
        # df_train, vectorizers = add_text_tokens_to_data_frame(df_train, text_train)
        # df_test, _ = add_text_tokens_to_data_frame(df_test, text_test, vectorizers=vectorizers)
        # model, _ = create_model(df_train, label_name, budget_classification)
        # print_model_evaluation(model, df_test, label_name, budget_classification)
    else:
        # treat outliers (no deletion because it changes target in test set as well)
        df_outl = treat_outliers_log_scale(data_frame.copy(), label_name=label_name, budget_name=label_name, add_to_df=False)

        # separate text
        data_frame, df_text = separate_text(data_frame)
        # separate text after outlier treatment
        df_outl, df_text_outl = separate_text(df_outl)

        print "\n\n##### With Outlier Treatment:"
        model, predictions, data_frame_target = create_model_cross_val(df_outl, label_name, budget_classification)
        if budget_classification:
            evaluate_classification(data_frame_target, predictions, label_name)
        else:
            evaluate_regression(data_frame_target, predictions, label_name)
        print_predictions_comparison(data_frame_target, predictions, label_name, 20)


def budget_model_production(connection, budget_name='budget', normalization=True):
    """ Learn model for label 'budget' on whole dataset and return it

    :param connection: RethinkDB connection
    :type connection: rethinkdb.net.ConnectionInstance
    :param budget_name: Predict 'total_charge' or 'budget'
    :type budget_name: str
    :param normalization: Whether to do min-max normalization
    :type normalization: bool
    """
    budget_classification = False

    data_frame = load_data_frame_from_db(connection)
    data_frame = prepare_data_budget_model(data_frame,
                                           budget_name, budget_classification)

    data_frame = treat_outliers_deletion(data_frame, budget_name=budget_name)
    data_frame, text_data = separate_text(data_frame, label_name=budget_name)
    data_frame, vectorizers = add_text_tokens_to_data_frame(data_frame, text_data)
    if normalization:
        data_frame, min, max = normalize_min_max(data_frame)
    else:
        min, max = [None, None]

    model, columns = create_model(data_frame, budget_name,
                                  is_classification=budget_classification,
                                  selectbest=False)

    return model, columns, min, max, vectorizers


def predict(data_frame, label_name, model, min=None, max=None):
    """ Predict budget for the given data frame

    :param data_frame: Pandas DataFrame holding the data for prediction
    :param label_name: Budget label
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


def budget_model_search(file_name):
    label_name = "budget"
    budget_classification = False
    do_cross_val = False
    # label_name = "total_charge"

    # data_frame = prepare_data(file_name)
    data_frame = load_data_frame_from_db()

    # prepare for model
    data_frame = prepare_data_budget_model(data_frame, label_name,
                                           budget_classification=budget_classification)
    # print_correlations(data_frame, label_name)

    if not do_cross_val:
        # split
        df_train, df_test = train_test_split(data_frame, train_size=0.8)

        # treat outliers
        df_train_outl, df_test_outl = treat_outliers(df_train.copy(),
                                                     df_test.copy(), label_name,
                                                     label_name, add_to_df=True)

        # separate text
        df_train, text_train = separate_text(df_train)
        df_test, text_test = separate_text(df_test)

        for k in range(1, len(df_train.columns)):
            result_collection = []
            for _ in range(1, len(df_train.columns)):
                model, relevant_indices = create_model(df_train.copy(),
                                                       label_name,
                                                       is_classification=False,
                                                       selectbest=k)
                results = print_model_evaluation(model, df_test.copy(), label_name,
                                                 is_classification=False, csv=True)
                results = results + list(relevant_indices)
                result_collection.append(results)

            abs_err_list = [sublist[1] for sublist in result_collection]
            best_results = abs_err_list.index(min(abs_err_list))
            print ",".join([str(item) for item in [label_name, k] + result_collection[best_results]])
    else:
        # treat outliers (no deletion because it changes target in test set as well)
        df_outl = treat_outliers_log_scale(data_frame.copy(),
                                           label_name=label_name,
                                           budget_name=label_name,
                                           add_to_df=False)

        # separate text
        data_frame, df_text = separate_text(data_frame)
        # separate text after outlier treatment
        df_outl, df_text_outl = separate_text(df_outl)

        print "\n\n##### With Outlier Treatment:"
        model, predictions, data_frame_target = create_model_cross_val(df_outl,
                                                                       label_name,
                                                                       budget_classification)
        if budget_classification:
            evaluate_classification(data_frame_target, predictions, label_name)
        else:
            evaluate_regression(data_frame_target, predictions, label_name)
        print_predictions_comparison(data_frame_target, predictions, label_name,
                                     20)
