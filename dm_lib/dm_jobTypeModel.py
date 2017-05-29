from dm_data_preparation import *
from dm_text_mining import do_text_mining
from dm_general import evaluate_classification, print_correlations
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def prepare_data_job_type_model(data_frame, label_name, relative_sampling):
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

    print "prepareDataJobTypeModel NOT IMPLEMENTED\n"

    # TODO just remove feedbacks?
    data_frame.dropna(subset=['feedback_for_client_availability', 'feedback_for_client_communication',
                              'feedback_for_client_cooperation', 'feedback_for_client_deadlines',
                              'feedback_for_client_quality', 'feedback_for_client_skills',
                              'feedback_for_freelancer_availability', 'feedback_for_freelancer_communication',
                              'feedback_for_freelancer_cooperation', 'feedback_for_freelancer_deadlines',
                              'feedback_for_freelancer_quality', 'feedback_for_freelancer_skills',
                              'budget'],
                      how='any', inplace=True)

    # drop columns where we don't have user data or are unnecessary
    drop_unnecessary = ["client_feedback", "client_past_hires"]
    data_frame.drop(labels=drop_unnecessary, axis=1, inplace=True)

    # TODO duplicate rows until fixed and hourly have the same ratio!
    # TODO or do weighting like in AADS
    min_target_value_count = min(data_frame[label_name].value_counts().values)
    print "Value counts:\n", \
        data_frame[label_name].value_counts().values, "\nminimum:", min_target_value_count,"\n ###\n"
    total_value_count = data_frame[label_name].value_counts().sum()
    fraction_hourly = float(len(data_frame.loc[data_frame[label_name] == 'Fixed'])) / total_value_count
    fraction_fixed = float(len(data_frame.loc[data_frame[label_name] == 'Hourly'])) / total_value_count

    if relative_sampling:
        sample_hourly = data_frame.ix[data_frame[label_name] == "Hourly"].sample(frac=fraction_hourly,
                                                                                 replace=False, random_state=0)
        sample_fixed = data_frame.ix[data_frame[label_name] == "Fixed"].sample(frac=fraction_fixed,
                                                                               replace=False, random_state=0)
    else:
        sample_hourly = data_frame.ix[data_frame[label_name] == "Hourly"].sample(n=min_target_value_count, replace=False,
                                                                                 random_state=0)
        sample_fixed = data_frame.ix[data_frame[label_name] == "Fixed"].sample(n=min_target_value_count, replace=False,
                                                                               random_state=0)
    data_frame = pd.concat([sample_hourly, sample_fixed])

    print "Value counts:\n", \
        data_frame[label_name].value_counts().values, "\nminimum:", min_target_value_count, "\n ###\n"

    ### convert everything to nominal
    data_frame, text_data = convert_to_nominal(data_frame, label_name)

    # print data_frame, "\n"
    print_data_frame("After preparing for budget model", data_frame)
    return data_frame, text_data


def job_type_model(file_name):
    """ Learn model for label 'job_type' and return it

    :param file_name: File name of JSON file containing the data
    :type file_name: str
    """
    label_name = "job_type"
    data_frame = prepare_data(file_name)
    data_frame, text_data = prepare_data_job_type_model(data_frame, label_name, relative_sampling=False)

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
