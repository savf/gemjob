from dm_data_preparation import *
from dm_text_mining import do_text_mining
from dm_general import evaluate_classification, print_correlations
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def prepare_data_job_type_model(data_frame, label_name):
    """ Prepare the given data to be used to predict the job type

    :param data_frame: Pandas DataFrame containing the data to be prepared
    :type data_frame: pandas.DataFrame
    :param label_name: Target label that will be predicted
    :type label_name: str
    :return: Cleaned Pandas DataFrames once with only nominal attributes and once only text attributes
    :rtype: pandas.DataFrame
    """

    # TODO duplicate rows until fixed and hourly have the same ratio!
    # or do weighting like in AADS

    # TODO remove unnecessary columns

    ### convert everything to nominal
    data_frame, text_data = convert_to_nominal(data_frame, label_name)
    print "prepareDataJobTypeModel NOT IMPLEMENTED"
    return data_frame, text_data


def job_type_model(file_name):
    """ Learn model for label 'job_type' and return it

    :param file_name: File name of JSON file containing the data
    :type file_name: str
    """
    label_name = "job_type"
    data_frame = prepare_data(file_name)
    data_frame, text_data = prepare_data_job_type_model(data_frame, label_name)

    print "\n\n########## Do Text Mining\n"
    text_train, text_test = train_test_split(text_data, train_size=0.8)
    do_text_mining(text_train, text_test, label_name, regression=False, max_features=5000)

    # print "\n\n########## Classification based on all data (except text)\n"
    # df_train, df_test = train_test_split(data_frame, train_size=0.8)
    #
    # forest = RandomForestClassifier(n_estimators=100)
    # forest.fit(df_train.ix[:, df_train.columns != label_name], df_train[label_name])
    # predictions = forest.predict(df_test.ix[:, df_train.columns != label_name])
    #
    # evaluate_classification(df_test, predictions, label_name)
    #
    # print "### Predictions: ###"
    # print predictions[0:8]
    # print "### Actual values: ###"
    # print df_test[label_name][0:8]
    # print "###########"
    #
    # print_correlations(data_frame, label_name)
