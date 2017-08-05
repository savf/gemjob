from sklearn.ensemble import BaggingRegressor, RandomForestClassifier, BaggingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, \
    mutual_info_regression, VarianceThreshold
from sklearn.model_selection import train_test_split

from dm_data_preparation import *
from dm_general import evaluate_classification, print_predictions_comparison, \
    generate_model_stats
from dm_text_mining import add_text_tokens_to_data_frame
from dm_feedbackModel import prepare_data_feedback_model, prepare_single_job_feedback_model, create_model

def prepare_data_feedback_model_classes(data_frame, label_name):
    """ Clean data specific to the feedback model with classes

    :param data_frame: Pandas DataFrame that holds the data
    :type data_frame: pandas.DataFrame
    :param label_name: Target label that will be predicted
    :type label_name: str
    :return: Cleaned Pandas DataFrames
    :rtype: pandas.DataFrame
    """
    data_frame = prepare_data_feedback_model(data_frame, label_name=label_name, do_balance_feedback=False)

    # print data_frame["feedback_for_client"].mean(), "\n"
    # print data_frame["feedback_for_client"][0:30], "\n"

    # print "MEAN =",data_frame["feedback_for_client"].mean()
    mean = 4.81
    data_frame.ix[data_frame[label_name] < mean, label_name] = 1
    data_frame.ix[data_frame[label_name] >= mean, label_name] = 2
    data_frame[label_name] = pd.cut(x=data_frame[label_name], bins=2,
                                    labels=["lower", "higher"])
    # data_frame.ix[(data_frame[label_name] <= 2), label_name] = 2
    # data_frame.ix[(data_frame[label_name] <= 3) & (data_frame[label_name] > 2), label_name] = 3
    # data_frame.ix[(data_frame[label_name] <= 4) & (data_frame[label_name] > 3), label_name] = 4
    # data_frame.ix[(data_frame[label_name] <= 5) & (data_frame[label_name] > 4), label_name] = 5
    # data_frame[label_name] = pd.cut(x=data_frame[label_name], bins=4,
    #                                 labels=["2stars", "3stars", "4stars", "5stars"])

    # print data_frame["feedback_for_client"].value_counts()

    data_frame = balance_data_set(data_frame, label_name, relative_sampling=False)

    # print data_frame["feedback_for_client"][0:30], "\n"
    # print data_frame["feedback_for_client"].value_counts()
    # print_data_frame("After preparing for feedback model", data_frame)

    return data_frame


def prepare_single_job_feedback_model_classes(data_frame, label_name,
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
    data_frame = prepare_single_job_feedback_model(data_frame, label_name,
                                      columns, min, max, vectorizers)

    mean = 4.81
    data_frame.ix[data_frame[label_name] < mean, label_name] = 1
    data_frame.ix[data_frame[label_name] >= mean, label_name] = 2
    data_frame[label_name] = pd.cut(x=data_frame[label_name], bins=2,
                                              labels=["lower", "higher"])

    return data_frame


def feedback_model_development_classes(file_name, connection=None):
    """ Learn model for label 'feedback' and return it

    :param file_name: JSON file containing all data
    :type file_name: str
    :param connection: RethinkDB connection to load the data (optional)
    :type connection: rethinkdb.net.ConnectionInstance
    """
    label_name = "feedback_for_client"
    feedback_classification = True

    #data_frame = prepare_data(file_name)
    data_frame = load_data_frame_from_db(connection)
    data_frame = prepare_data_feedback_model_classes(data_frame, label_name)

    df_train, df_test = train_test_split(data_frame, train_size=0.8)

    # df_test = balance_data_set(df_test, label_name, relative_sampling=False)

    df_train, df_train_text = separate_text(df_train, label_name)
    df_test, df_test_text = separate_text(df_test, label_name)

    df_train, vectorizers = add_text_tokens_to_data_frame(df_train,
                                                          df_train_text)

    df_test, _ = add_text_tokens_to_data_frame(df_test, df_test_text,
                                               vectorizers=vectorizers)

    model, columns = create_model(df_train, label_name, feedback_classification, selectbest=False, variance_threshold=True)
    predictions = model.predict(df_test[columns])

    # print_predictions_comparison(df_test, predictions, label_name, 50)

    return evaluate_classification(df_test[label_name], predictions, label_name)


def predict_classes(data_frame, label_name, model, min=None, max=None):
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


def feedback_model_classes_production(connection, label_name='feedback_for_client',
                              normalization=False):
    """ Learn model for label 'client_feedback' on whole dataset and return it
    
    :param connection: RethinkDB connection
    :type connection: rethinkdb.net.ConnectionInstance
    :param label_name: Feedback label to predict (default 'client_feedback')
    :type label_name: str
    :param normalization: Whether to do min-max normalization
    :type normalization: bool
    """
    feedback_classification = True

    data_frame = load_data_frame_from_db(connection)
    data_frame = prepare_data_feedback_model_classes(data_frame, label_name)

    data_frame, text_data = separate_text(data_frame, label_name=label_name)
    data_frame, vectorizers = add_text_tokens_to_data_frame(data_frame,
                                                            text_data)
    if normalization:
        data_frame, min, max = normalize_min_max(data_frame)
    else:
        min, max = [None, None]

    model, columns = create_model(data_frame, label_name,
                                  feedback_classification,
                                  selectbest=False,
                                  variance_threshold=True)

    importances = generate_model_stats(data_frame[columns], model)

    return model, columns, min, max, vectorizers, importances
