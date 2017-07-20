from dm_clustering import *

def predict_knn(unnormalized_data, normalized_train, normalized_test, k, target_columns):

    # get actual target_columns (dummies)
    actual_cols = []
    for tc in target_columns:
        actual_cols = actual_cols + [col for col in list(normalized_train) if col.startswith(tc)]

    # drop target columns in train and in test data so distance not based on them
    normalized_train.drop(labels=actual_cols, axis=1, inplace=True)
    normalized_test.drop(labels=actual_cols, axis=1, inplace=True)

    # get neighbors based on euclidean distance
    distances_dict = {}
    for index_test, row_test in normalized_test.iterrows():
        distances = euclidean_distances(normalized_train, row_test.values.reshape(1, -1))
        distances = pd.DataFrame(distances, columns=["distances"], index=normalized_train.index)
        distances.sort_values(by="distances", axis=0, inplace=True)
        # get k nearest
        distances_dict[index_test] = distances[0:k]

    print "\n\n### Predictions based on", k, " nearest neighbors: \n"
    numeric_columns = unnormalized_data._get_numeric_data().columns

    for tc in target_columns:

        if tc in numeric_columns:
            unnormalized_data[tc+"_prediction"] = 0
        else:
            unnormalized_data[tc + "_prediction"] = ""

        # print "\n\n\n\n##### Predict label:", tc
        correct_predict = 0
        abs_err_predict = 0

        for index_test, row_test in normalized_test.iterrows():
            # print "\n#### Current row:", index_test
            actual = unnormalized_data.loc[index_test][tc]
            # print "## Actucal value:", actual
            # print "## Neighbor values:"
            neighbor_indices = distances_dict[index_test].index.values
            neighbor_values = unnormalized_data.ix[neighbor_indices]

            if tc in numeric_columns:
                median = neighbor_values[tc].median()
                abs_err = abs(actual - median)
                abs_err_predict = abs_err_predict + abs_err
                # print "# Median:", median, "Error:", abs_err
                unnormalized_data.set_value(index_test, tc + "_prediction", median)
            else:
                value_counts = neighbor_values[tc].value_counts()
                if len(value_counts) > 0:
                    majority = value_counts.idxmax(axis=1)
                else:
                    majority = np.NaN
                if majority == actual:
                    correct_predict = correct_predict + 1
                # print "Majority voting:", majority
                unnormalized_data.set_value(index_test, tc + "_prediction", majority)


        if tc in numeric_columns:
            print "### Abs Error:", abs_err_predict / float(normalized_test.shape[0])
        else:
            print "### Correctly predicted:", correct_predict, "in %:", float(correct_predict) / float(normalized_test.shape[0])
        print "### Number of test rows:", normalized_test.shape[0]

        return unnormalized_data.loc[normalized_test.index]


def test_knn(file_name, target="budget"):
    """ Test k nearest neighbors for predictions (with test and train set)

    :param file_name: JSON file containing all data
    :type file_name: str
    :param target: Target label to predict
    :type target: str
    """

    data_frame = prepare_data(file_name)
    data_frame_original = data_frame.copy()

    df_train, df_test = train_test_split(data_frame, train_size=0.8)

    # in app, train data would be prepared in db
    df_train, min, max, vectorizers = prepare_data_clustering(df_train, z_score_norm=False, add_text=True)

    if target == "budget":
        # remove rows without budget to predict_comparison budget
        df_test.ix[df_test.job_type == 'Hourly', 'budget'] = None
        df_test.dropna(subset=["budget"], how='any', inplace=True)
    elif target == "job_type" or target == "experience_level" or target == "subcategory2":
        df_test = balance_data_set(df_test, target, relative_sampling=False)

    # prepare test data
    df_test = prepare_test_data_clustering(df_test, df_train.columns, min, max, vectorizers=vectorizers, weighting=True)

    unnormalized_data = predict_knn(data_frame_original.copy(), df_train.copy(), df_test.copy(), k=15, target_columns=[target])

    print "\n"
    if target == "budget":
        evaluate_regression(unnormalized_data[target], unnormalized_data[target+'_prediction'], target)
    elif target == "job_type" or target == "experience_level" or target == "subcategory2":
        evaluate_classification(unnormalized_data[target], unnormalized_data[target+'_prediction'], target)
