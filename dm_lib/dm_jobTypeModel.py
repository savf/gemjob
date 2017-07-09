from dm_data_preparation import *
from dm_text_mining import do_text_mining
from dm_general import evaluate_classification, print_predictions_comparison
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import tree

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

    # drop columns where we don't have user data or are unnecessary
    drop_unnecessary = ["client_feedback", "client_past_hires"]
    data_frame.drop(labels=drop_unnecessary, axis=1, inplace=True)

    # balance data set so ratio of hourly and fixed is 1:1
    data_frame = balance_data_set(data_frame, label_name, relative_sampling=relative_sampling)

    # fill missing experience levels with random non-missing values
    filled_experience_levels = data_frame["experience_level"].dropna()
    data_frame["experience_level"] = data_frame.apply(
        lambda row: row["experience_level"] if row["experience_level"] is not None
        else random.choice(filled_experience_levels), axis=1)

    # TODO convert everything to numeric? need that for quite a lot of classifiers
    data_frame = convert_to_numeric(data_frame, label_name)
    ### roughly cluster by rounding
    # data_frame = coarse_clustering(data_frame, label_name)

    # print data_frame, "\n"
    print_data_frame("After preparing for job type model", data_frame)
    return separate_text(data_frame, label_name)


def job_type_model(file_name):
    """ Learn model for label 'job_type' and return it

    :param file_name: File name of JSON file containing the data
    :type file_name: str
    """
    label_name = "job_type"

    #data_frame = prepare_data(file_name)
    data_frame = load_data_frame_from_db()
    data_frame, text_data = prepare_data_job_type_model(data_frame, label_name, relative_sampling=False)

    # print "\n\n########## Do Text Mining\n"
    # text_train, text_test = train_test_split(text_data, train_size=0.8)
    # do_text_mining(text_train, text_test, label_name, regression=False, max_features=5000)

    print "\n\n########## Classification based on all data (except text)\n"
    df_train, df_test = train_test_split(data_frame, train_size=0.8)

    #clf = SVC(kernel='linear')
    #clf = RandomForestClassifier(n_estimators=100)
    clf = tree.DecisionTreeClassifier()

    clf.fit(df_train.ix[:, df_train.columns != label_name], df_train[label_name])
    predictions = clf.predict(df_test.ix[:, df_test.columns != label_name])

    evaluate_classification(df_test, predictions, label_name)

    print_predictions_comparison(df_test, predictions, label_name)

    #with open("job_type_tree.dot", 'w') as f:
    #    f = tree.export_graphviz(clf, feature_names=df_train.columns.values, out_file=f)

    # print_correlations(data_frame, label_name)
