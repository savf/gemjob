from dm_clustering import *

def predict_knn(unnormalized_data_predict, unnormalized_data_train, normalized_predict, normalized_train, k, target_columns, do_reweighting=True):
    drop_unnecessary = ["skills", "title", "snippet", "client_country", "date_created", "client_reviews_count"]
    unnormalized_data_predict.drop(labels=drop_unnecessary, axis=1, inplace=True)

    # get actual target_columns (dummies)
    actual_cols = []
    for tc in target_columns:
        actual_cols = actual_cols + [col for col in list(normalized_train) if col.startswith(tc)]

    numeric_columns = unnormalized_data_train._get_numeric_data().columns

    # drop target columns in train and in test data so distance not based on them
    normalized_train.drop(labels=actual_cols, axis=1, inplace=True)
    normalized_predict.drop(labels=actual_cols, axis=1, inplace=True)

    # setup prediction attributes
    unnormalized_data_predict["title_prediction"] = ""
    unnormalized_data_predict["snippet_prediction"] = ""
    for tc in target_columns:
        if tc in numeric_columns:
            unnormalized_data_predict[tc + "_prediction"] = 0
        else:
            unnormalized_data_predict[tc + "_prediction"] = ""

    # get neighbors based on euclidean distance
    # distances_dict = {}
    for index_test, row_test in normalized_predict.iterrows():
        row_df = pd.DataFrame(row_test.values.reshape(1, -1), index=[0], columns=list(normalized_train.columns))
        if do_reweighting:
            row_df, normalized_train_rw = reduce_tokens_to_single_job(row_df.copy(), normalized_train.copy())
        else:
            normalized_train_rw = normalized_train
        distances = euclidean_distances(normalized_train_rw, row_df)
        distances = pd.DataFrame(distances, columns=["distances"], index=normalized_train_rw.index)
        distances.sort_values(by="distances", axis=0, inplace=True)
        # get k nearest
        # distances_dict[index_test] = distances[0:k]
        neighbor_indices = distances[0:k].index.values
        neighbor_values = unnormalized_data_train.ix[neighbor_indices]

        unnormalized_data_predict.set_value(index_test, "title_prediction", neighbor_values["title"][0])
        unnormalized_data_predict.set_value(index_test, "snippet_prediction", neighbor_values["snippet"][0])

        # print "\n\n ######## Title ######## \n"
        # print neighbor_values["title"][0]
        # print "\n\n ######## Title Predicted ######## \n"
        # print unnormalized_data_predict.loc[index_test]["title"]

        for tc in target_columns:

            # if tc in numeric_columns:
            #     unnormalized_data_predict[tc+"_prediction"] = 0
            # else:
            #     unnormalized_data_predict[tc + "_prediction"] = ""
            # # print "\n\n\n\n##### Predict label:", tc
            # correct_predict = 0
            # abs_err_predict = 0
            #
            # for index_test, row_test in normalized_predict.iterrows():
                # print "\n#### Current row:", index_test
                # actual = unnormalized_data_predict.loc[index_test][tc]
                # print "## Actucal value:", actual
                # print "## Neighbor values:"

            # neighbor_indices = distances_dict[index_test].index.values
            # neighbor_values = unnormalized_data_train.ix[neighbor_indices]

            if tc in numeric_columns:
                median = neighbor_values[tc].median()
                # abs_err = abs(actual - median)
                # abs_err_predict = abs_err_predict + abs_err
                # print "# Median:", median, "Error:", abs_err
                unnormalized_data_predict.set_value(index_test, tc + "_prediction", median)
            else:
                value_counts = neighbor_values[tc].value_counts()
                if len(value_counts) > 0:
                    majority = value_counts.idxmax(axis=1)
                else:
                    majority = np.NaN
                # if majority == actual:
                #     correct_predict = correct_predict + 1
                # print "Majority voting:", majority
                unnormalized_data_predict.set_value(index_test, tc + "_prediction", majority)

            # if tc in numeric_columns:
            #     print "### Abs Error:", abs_err_predict / float(normalized_predict.shape[0])
            # else:
            #     print "### Correctly predicted:", correct_predict, "in %:", float(correct_predict) / float(normalized_predict.shape[0])
            # print "### Number of test rows:", normalized_predict.shape[0]

    return unnormalized_data_predict


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
    df_train, min, max, vectorizers = prepare_data_clustering(df_train, z_score_norm=False, add_text=True, do_log_transform=True)

    if target == "budget":
        # remove rows without budget to predict_comparison budget
        df_test.ix[df_test.job_type == 'Hourly', 'budget'] = None
        df_test.dropna(subset=["budget"], how='any', inplace=True)
    elif target == "job_type" or target == "experience_level" or target == "subcategory2":
        df_test = balance_data_set(df_test, target, relative_sampling=False)

    # prepare test data
    df_test = prepare_test_data_clustering(df_test, df_train.columns, min, max, vectorizers=vectorizers, weighting=True, do_log_transform=True)

    # Reweighting harms budget! Benefits subcategory however and shows more similar example snippet and title
    unnormalized_data = predict_knn(data_frame_original.loc[df_test.index], data_frame_original.loc[df_train.index], df_test.copy(), df_train.copy(), k=15, target_columns=[target], do_reweighting=False)

    print "\n"
    if target in ["budget", "client_feedback", "feedback_for_client", "feedback_for_freelancer"]:
        evaluate_regression(unnormalized_data[target], unnormalized_data[target+'_prediction'], target)
    elif target == "job_type" or target == "experience_level" or target == "subcategory2":
        evaluate_classification(unnormalized_data[target], unnormalized_data[target+'_prediction'], target)
