from dm_data_preparation import *
from dm_text_mining import do_text_mining
from sklearn.model_selection import train_test_split


def prepare_data_job_type_model(data_frame):
    """ Prepare the given data to be used to predict the job type

    :param data_frame: Pandas DataFrame containing the data to be prepared
    :type data_frame: pandas.DataFrame
    :return: Cleaned data frame
    :rtype: pandas.DataFrame
    """
    # TODO
    ### convert everything to nominal
    convert_to_nominal(data_frame)
    print "prepareDataJobTypeModel NOT IMPLEMENTED"
    return data_frame


def job_type_model(file_name):
    """ Learn model for label 'job_type' and return it

    :param file_name: File name of JSON file containing the data
    :type file_name: str
    """
    label_name = "job_type"
    data_frame = prepare_data(file_name)
    data_frame = prepare_data_job_type_model(data_frame)

    df_train, df_test = train_test_split(data_frame, train_size=0.8)

    do_text_mining(df_train, df_test, label_name, regression=False, max_features=5000)

    # printCorr(data_frame, label_name)
