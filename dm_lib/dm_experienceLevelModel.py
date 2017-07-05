from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from dm_data_preparation import *
from dm_general import evaluate_classification, print_predictions_comparison


def prepare_data_experience_level_model(data_frame, label_name, relative_sampling):
    """
    Prepare data to be used to precict the experience level

    :param data_frame: Pandas DataFrame containing the data
    :param label_name: Target label
    :param relative_sampling: Relative or 1:1 sampling
    :type relative_sampling: bool
    :return: DataFrame prepared for experience_level classification
    :rtype: pandas.DataFrame
    """
    data_frame.dropna(subset=['experience_level'], inplace=True)

    data_frame = convert_to_numeric(data_frame, label_name)
    data_frame = balance_data_set(data_frame, label_name, relative_sampling)

    # print data_frame
    print_data_frame("After preparing for budget model", data_frame)

    return data_frame


def experience_level_model(file_name):
    """
    Learn model for the experience level label and return it

    :param file_name: JSON file containing all data
    :return:
    """

    label_name = 'experience_level'

    #data_frame = prepare_data(file_name)
    data_frame = load_data_frame_from_db()
    data_frame = prepare_data_experience_level_model(data_frame, label_name,
                                                     relative_sampling=False)
    data_frame, text_data = separate_text(data_frame, label_name)

    df_train, df_test = train_test_split(data_frame, train_size=0.8)

    clf = SVC(kernel='linear')
    # clf = RandomForestClassifier(n_estimators=100)

    clf.fit(df_train.ix[:, df_train.columns != label_name], df_train[label_name])
    predictions = clf.predict(df_test.ix[:, df_test.columns != label_name])

    evaluate_classification(df_test, predictions, label_name)

    print_predictions_comparison(df_test, predictions, label_name, 50)
