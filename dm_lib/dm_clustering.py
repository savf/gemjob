from dm_data_preparation import *
from dm_general import *
from sklearn.cluster import DBSCAN, KMeans, MeanShift, estimate_bandwidth
from sklearn import metrics
import numpy as np
from dm_text_mining import addTextTokensToWholeDF
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances

def prepare_data_clustering(data_frame, z_score_norm=False, add_text=False, weighting=True):
    """ Clean and prepare data specific to clustering

    :param data_frame: Pandas DataFrame that holds the data
    :type data_frame: pandas.DataFrame
    :param z_score_norm: Use z-score normalization
    :type z_score_norm: bool
    :param add_text: Add text tokens
    :type add_text: bool
    :param weighting: Do weighting
    :type weighting: bool
    :return: Cleaned Pandas DataFrame with only numerical attributes and either min/max or mean/std, if text added also vectorizers
    :rtype: pandas.DataFrame
    """

    # drop columns that are unnecessary for clustering: are they?
        # idea: we don't want to predict anymore, we just want to cluster based on interesting attributes provided by user
    drop_unnecessary = ["date_created", "client_jobs_posted", "client_past_hires", "client_reviews_count"]
    data_frame.drop(labels=drop_unnecessary, axis=1, inplace=True)

    # declare total_charge as missing, if 0
    data_frame.ix[data_frame.total_charge == 0, 'total_charge'] = None

    # rows that don't contain total_charge
    data_frame.dropna(subset=["total_charge"], how='any', inplace=True)

    # overall feedbacks
    data_frame = get_overall_job_reviews(data_frame)

    # declare feedbacks as missing, if 0
    data_frame.ix[data_frame.client_feedback == 0, 'client_feedback'] = None

    # remove rows with missing values

    # feedbacks
    data_frame['feedback_for_client'].fillna(data_frame['feedback_for_client'].mean(), inplace=True)
    data_frame['feedback_for_freelancer'].fillna(data_frame['feedback_for_freelancer'].mean(), inplace=True)
    data_frame['client_feedback'].fillna(data_frame['client_feedback'].mean(), inplace=True)

    # fill missing experience levels with random non-missing values
    # filled_experience_levels = data_frame["experience_level"].dropna()
    # data_frame["experience_level"] = data_frame.apply(
    #     lambda row: row["experience_level"] if row["experience_level"] is not None
    #     else random.choice(filled_experience_levels), axis=1)

    # convert everything to numeric
    data_frame = convert_to_numeric(data_frame, label_name="")

    # handle text
    data_frame, text_data = separate_text(data_frame)
    if add_text:
        data_frame, vectorizers = addTextTokensToWholeDF(data_frame, text_data)
    else:
        vectorizers = {}

    # normalize
    if z_score_norm:
        data_frame, mean, std = normalize_z_score(data_frame)
    else:
        data_frame, min, max = normalize_min_max(data_frame)

    if weighting:
        data_frame = weight_data(data_frame)

    # print data_frame[0:5], "\n"
    # print_data_frame("After preparing for clustering", data_frame)
    # print_statistics(data_frame)

    if z_score_norm:
        return data_frame, mean, std, vectorizers
    else:
        return data_frame, min, max, vectorizers


def prepare_test_data(data_frame, cluster_columns, min, max, vectorizers=None, weighting=True):
    """ Clean and prepare data specific to clustering

    :param data_frame: Pandas DataFrame that holds the data
    :type data_frame: pandas.DataFrame
    :param cluster_columns: Columns of cluster data
    :type cluster_columns: list
    :param min: minimum values
    :type min: pandas.Series
    :param max: maximum values
    :type max: pandas.Series
    :param vectorizers: Vectorizers for adding text tokens (text not added if not given!)
    :type vectorizers: dict of sklearn.feature_extraction.text.CountVectorizer
    :param weighting: Do weighting
    :type weighting: bool
    :return: Cleaned Pandas DataFrames once with only numerical attributes and once only text attributes
    :rtype: pandas.DataFrame
    """

    # drop columns that are unnecessary for clustering: are they?
        # idea: we don't want to predict anymore, we just want to cluster based on interesting attributes provided by user
    drop_unnecessary = ["date_created", "client_jobs_posted", "client_past_hires", "client_reviews_count"]
    data_frame.drop(labels=drop_unnecessary, axis=1, inplace=True)

    # declare total_charge as missing, if 0
    data_frame.ix[data_frame.total_charge == 0, 'total_charge'] = None

    # rows that don't contain total_charge
    data_frame.dropna(subset=["total_charge"], how='any', inplace=True)

    # overall feedbacks
    data_frame = get_overall_job_reviews(data_frame)

    # declare feedbacks as missing, if 0
    data_frame.ix[data_frame.client_feedback == 0, 'client_feedback'] = None

    # remove rows with missing values

    # feedbacks
    data_frame['feedback_for_client'].dropna(how='any', inplace=True)
    data_frame['feedback_for_freelancer'].dropna(how='any', inplace=True)
    data_frame['client_feedback'].dropna(how='any', inplace=True)

    # experience level
    # data_frame["experience_level"].dropna(how='any', inplace=True)

    # convert everything to numeric
    data_frame = convert_to_numeric(data_frame, label_name="")

    # handle text
    data_frame, text_data = separate_text(data_frame)
    if vectorizers is not None:
        data_frame, _ = addTextTokensToWholeDF(data_frame, text_data, vectorizers=vectorizers)

    # add missing columns (dummies, that were not in this data set)
    for col in cluster_columns:
        if col not in data_frame.columns:
            data_frame[col] = 0
    # remove columns not existing in clusters
    for col in data_frame.columns:
        if col not in cluster_columns:
            data_frame.drop(labels=[col], axis=1, inplace=True)
    # normalize
    data_frame, _, _ = normalize_min_max(data_frame, min, max)

    if weighting:
        data_frame = weight_data(data_frame)

    # order acording to cluster_columns (important!!! scikit does not look at labels!)
    # print data_frame[0:5], "\n"
    data_frame = data_frame.reindex_axis(cluster_columns, axis=1)
    # print "\n\n\n###############################\n\n\n"
    # print data_frame[0:5], "\n"
    # print_data_frame("After preparing for clustering", data_frame)
    # print_statistics(data_frame)

    return data_frame


def explore_clusters(clusters, original_data_frame, silhouette_score, name=""):
    """ Print stats and facts about the clusters

    :param clusters: List of clusters in the form of Pandas Data Frames
    :type clusters: list
    """
    selected_nominal_colums = ['client_country', 'experience_level', 'job_type', 'subcategory2']
    selected_numeric_colums = ['duration_weeks_median', 'duration_weeks_total', 'client_feedback',
                               'feedback_for_client', 'feedback_for_freelancer', 'total_charge', 'skills_number',
                               'snippet_length']

    print "\n\n\n#################### Explore clusters: " + name + " ####################\n"
    print "Number of clusters:", len(clusters)

    avg_unique_vals = {}
    for nom_col in selected_nominal_colums:
        avg_unique_vals[nom_col] = 0

    avg_std_deviations = {}
    # all_means = {}
    for num_col in selected_numeric_colums:
        avg_std_deviations[num_col] = 0
        # all_means[num_col] = []

    # TODO: store stats for each cluster into a file
    for cluster in clusters:
        print "\n\nCluster: " + str(cluster["cluster_label"][0]), " --- Shape: ", cluster.shape
        # print '\033[94m', cluster[0:5], '\033[0m'

        for num_col in selected_numeric_colums:
            mean = cluster[num_col].mean()
            # all_means[num_col].append(mean)

            std = cluster[num_col].std()
            if np.isnan(std):
                # all values are nan
                std = 0
                mean = 0
            avg_std_deviations[num_col] = avg_std_deviations[num_col] + std

            print num_col, " --- mean:", mean, ", std:", std

        for nom_col in selected_nominal_colums:
            unique = cluster[nom_col].unique()
            val_counts = cluster[nom_col].value_counts()

            print nom_col, " --- unique values:", len(unique)
            avg_unique_vals[nom_col] = avg_unique_vals[nom_col] + len(unique)
            # print '\033[94m\n', val_counts, '\033[0m\n'

    print "\n\n####### " + name + " - Overall analysis: #######"
    print "Number of clusters:", len(clusters)
    print "Silhouette Coefficient: ", silhouette_score, "\n" # http://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient

    # - average of unique values per nominal column
    nom_col_score = 0
    for nom_col in selected_nominal_colums:
        total_unique =  len(original_data_frame[nom_col].unique())
        avg_unique_vals[nom_col] = float(avg_unique_vals[nom_col]) / float(len(clusters)) # average unqiue
        print nom_col, " --- avg unique per cluster:", avg_unique_vals[nom_col], "of total unqiue:", total_unique
        nom_col_score = nom_col_score + avg_unique_vals[nom_col] / total_unique # for final score, divide by total unique

    print "### Total score nominal (lower is better):", nom_col_score, "\n"


    # - average std deviation and std deviation of means for each numerical column
            # -> how different are means between clusters, how similar is data within cluster (std)
    num_col_score = 0
    for num_col in selected_numeric_colums:
        avg_std_deviations[num_col] = avg_std_deviations[num_col] / float(len(clusters))
        # std_of_means = np.std(np.array(all_means[num_col]))
        total_std = original_data_frame[num_col].std()
        print num_col, " --- average std per cluster:", avg_std_deviations[num_col], ", std of original df:", total_std
        num_col_score = num_col_score + avg_std_deviations[num_col] / total_std  # for final score, divide by total std

    print "### Total score numerical (lower is better):", num_col_score

    # generate final score to easily compare all clustering algorithms and normalizations
        # more or less based on homogenity of clusters (but not completeness) http://scikit-learn.org/stable/modules/clustering.html#homogeneity-completeness-and-v-measure
    final_score = num_col_score + nom_col_score
    print "\n### Final score (lower is better):", final_score

    print "\n##########################################################\n"

    return final_score


def do_clustering_dbscan(data_frame, find_best_params=False, explore_clusters=True):
    """ Cluster using DBSCAN algorithm
    silhouette_score about 0.58 without removing columns
    silhouette_score about 0.64 WITH removing columns

    :param data_frame: Pandas DataFrame holding non-normalized data
    :type data_frame: pandas.DataFrame
    :param find_best_params: Find best parameters for clustering
    :type find_best_params: bool
    :param explore_clusters: Print stats about clusters
    :type explore_clusters: bool
    """

    min_n_clusters = 10
    max_n_clusters = 500

    data_frame_original = get_overall_job_reviews(data_frame.copy())

    # prepare for clustering
    data_frame, min, max, vectorizers = prepare_data_clustering(data_frame, z_score_norm=False, add_text=True)
    # print data_frame[0:5]

    if find_best_params:
        best_silhouette_score = -1000
        best_eps = -1
        best_min_samples = -1
        best_n_clusters = -1
        config_num = 1
        for eps in np.arange(0.7, 60, 0.1):
            for min_samples in range(2, 4, 1):
                print "\n## Config ", config_num, "---  eps=", eps, "; min_samples=", min_samples, " ##"
                config_num = config_num+1

                db = DBSCAN(eps=eps, min_samples=min_samples).fit(data_frame)
                labels = db.labels_

                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                print "Number of clusters: ", n_clusters
                if n_clusters > min_n_clusters and n_clusters < max_n_clusters:
                    silhouette_score = metrics.silhouette_score(data_frame, labels)
                    print "Silhouette Coefficient: ", silhouette_score

                    if silhouette_score > best_silhouette_score:
                        best_silhouette_score = silhouette_score
                        best_eps = eps
                        best_min_samples = min_samples
                        best_n_clusters = n_clusters
                        print "!New Best!"
                elif n_clusters <= min_n_clusters:
                    break
        print "\n ### Result: "
        print "best_eps=", best_eps, "; best_min_samples=", best_min_samples, "; best_n_clusters=", best_n_clusters, "; best_silhouette_score=", best_silhouette_score
    else:
        # db = DBSCAN(eps=38.5, min_samples=2).fit(data_frame) # weighted, z-score
        db = DBSCAN(eps=1.0, min_samples=2).fit(data_frame)  # no text, min-max
        labels = db.labels_
        centroids = pd.DataFrame(db.cluster_centers_, columns=data_frame.columns)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print "Number of clusters: ", n_clusters
        silhouette_score = metrics.silhouette_score(data_frame, labels)

        # cluster the original data frame
        data_frame["cluster_label"] = labels
        data_frame = data_frame[data_frame.cluster_label != -1] # remove noisy samples (have cluster label -1)

        data_frame_original = data_frame_original.join(data_frame["cluster_label"], how='inner')

        gb = data_frame_original.groupby('cluster_label')
        clusters = [gb.get_group(x) for x in gb.groups]

        if explore_clusters:
            explore_clusters(clusters, data_frame_original, silhouette_score, "DBSCAN")

        return db, clusters, centroids, min, max, vectorizers


def do_clustering_kmeans(data_frame, find_best_params=False, explore_clusters=True):
    """ Cluster using k-means algorithm
    silhouette_score about 0.54 (0.25 with z-score) without removing columns
    silhouette_score about 0.60 WITH removing columns

    :param data_frame: Pandas DataFrame holding non-normalized data
    :type data_frame: pandas.DataFrame
    :param find_best_params: Find best parameters for clustering
    :type find_best_params: bool
    :param explore_clusters: Print stats about clusters
    :type explore_clusters: bool
    """

    min_n_clusters = 10
    max_n_clusters = 500

    data_frame_original = get_overall_job_reviews(data_frame.copy())

    # prepare for clustering
    data_frame, min, max, vectorizers = prepare_data_clustering(data_frame, z_score_norm=False, add_text=True)
    # print data_frame[0:5]

    if find_best_params:
        best_silhouette_score = -1000
        best_k = -1
        config_num = 1

        for k in range(40, max_n_clusters, 5): # ran it until 250, began to get worse after 98
            print "\n## Config ", config_num, "---  k=", k, " ##"
            config_num = config_num+1

            kmeans = KMeans(n_clusters=k).fit(data_frame)
            labels = kmeans.labels_

            silhouette_score = metrics.silhouette_score(data_frame, labels)
            print "Silhouette Coefficient: ", silhouette_score

            if silhouette_score > best_silhouette_score:
                best_silhouette_score = silhouette_score
                best_k = k
                print "!New Best!"
        print "\n ### Result: "
        print "best_k=", best_k, "; best_silhouette_score=", best_silhouette_score
    else:
        # kmeans = KMeans(n_clusters=98).fit(data_frame) # without text, min-max scaling
        kmeans = KMeans(n_clusters=93).fit(data_frame)  # with text, min-max scaling
        labels = kmeans.labels_
        centroids = pd.DataFrame(kmeans.cluster_centers_, columns=data_frame.columns)

        silhouette_score = metrics.silhouette_score(data_frame, labels)

        # cluster the original data frame
        data_frame["cluster_label"] = labels
        data_frame = data_frame[data_frame.cluster_label != -1] # remove noisy samples (have cluster label -1)

        data_frame_original = data_frame_original.join(data_frame["cluster_label"], how='inner')

        gb = data_frame_original.groupby('cluster_label')
        clusters = [gb.get_group(x) for x in gb.groups]

        if explore_clusters:
            explore_clusters(clusters, data_frame_original, silhouette_score, "K-Means")

        return kmeans, clusters, centroids, min, max, vectorizers


def do_clustering_mean_shift(data_frame, find_best_params=False, explore_clusters=True):
    """ Cluster using mean-shift algorithm
    silhouette_score about 0.58 without removing columns
    silhouette_score about 0.65 WITH removing columns
    silhouette_score about 0.85 with removing columns WITH z-score BUT only 2 clusters

    :param data_frame: Pandas DataFrame holding non-normalized data
    :type data_frame: pandas.DataFrame
    :param find_best_params: Find best parameters for clustering
    :type find_best_params: bool
    :param explore_clusters: Print stats about clusters
    :type explore_clusters: bool
    """

    data_frame_original = get_overall_job_reviews(data_frame.copy())

    # prepare for clustering
    data_frame, min, max, vectorizers = prepare_data_clustering(data_frame, z_score_norm=False, add_text=True)
    # print data_frame[0:5]

    if find_best_params:
        best_silhouette_score = -1000
        best_bandwidth = -1
        config_num = 1

        for bandwidth in np.arange(0.3, 2.0, 0.1):
            # bandwidth = estimate_bandwidth(data_frame, quantile=q) # doesn't work, probably too much data
            print "\n## Config ", config_num, "---  bandwidth=", bandwidth, " ##"
            config_num = config_num+1

            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(data_frame)
            labels = ms.labels_

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            print "Number of clusters: ", n_clusters
            silhouette_score = metrics.silhouette_score(data_frame, labels)
            print "Silhouette Coefficient: ", silhouette_score

            if silhouette_score > best_silhouette_score:
                best_silhouette_score = silhouette_score
                best_bandwidth = bandwidth
                best_n_clusters = n_clusters
                print "!New Best!"
        print "\n ### Result: "
        print "best_bandwidth=", best_bandwidth, "; best_n_clusters=", best_n_clusters, "; best_silhouette_score=", best_silhouette_score
    else:
        ms = MeanShift(bandwidth=0.9, bin_seeding=True).fit(data_frame) # min-max
        # ms = MeanShift(bandwidth=38.0, bin_seeding=True).fit(data_frame) # z-score
        labels = ms.labels_
        centroids = pd.DataFrame(ms.cluster_centers_, columns=data_frame.columns)

        silhouette_score = metrics.silhouette_score(data_frame, labels)

        # cluster the original data frame
        data_frame["cluster_label"] = labels
        data_frame = data_frame[data_frame.cluster_label != -1] # remove noisy samples (have cluster label -1)

        data_frame_original = data_frame_original.join(data_frame["cluster_label"], how='inner')

        gb = data_frame_original.groupby('cluster_label')
        clusters = [gb.get_group(x) for x in gb.groups]

        if explore_clusters:
            explore_clusters(clusters, data_frame_original, silhouette_score, "Mean-Shift")


        # # See if we get correct clusters for training set when removing a column
        # cols_to_set_0 = [col for col in list(data_frame) if col.startswith("experience_level")]
        # data_frame[cols_to_set_0] = 0
        # for index, row in data_frame.drop(labels=["cluster_label"], axis=1)[0:10].iterrows():
        #     distances = euclidean_distances(centroids.drop(labels=cols_to_set_0, axis=1), row.drop(labels=cols_to_set_0).values.reshape(1, -1))
        #     cluster_index = np.array(distances).argmin()
        #     print "Predicted:", ms.predict(row.values.reshape(1, -1))[0]
        #     print "Distance based:", cluster_index
        #     print "Actual:", data_frame.loc[index]["cluster_label"]

        return ms, clusters, centroids, min, max, vectorizers


def predict(model, unnormalized_data, normalized_data, clusters, centroids, target_columns):
    """ Predict columns based on clusters

    :param model: Trained cluster model
    :type model: sklearn.cluster.MeanShift or similar
    :param unnormalized_data: Pandas DataFrames holding non-normalized data
    :type unnormalized_data: pandas.DataFrame
    :param normalized_data: Pandas DataFrames holding normalized data
    :type normalized_data: pandas.DataFrame
    :param clusters: List of Pandas DataFrames holding non-normalized data
    :type clusters: pandas.DataFrame
    :param centroids: minimum values
    :type centroids: pandas.DataFrame
    :param target_columns: columns to predict
    :type target_columns: list
    """
    predicted_the_same = 0

    # get actual target_columns (dummies)
    actual_cols = []
    for tc in target_columns:
        actual_cols = actual_cols + [col for col in list(normalized_data) if col.startswith(tc)]

    # TODO: "predict" predicts always the same cluster for every row -> why?
    # set columns to predict to 0
    normalized_data[actual_cols] = 0
    # use model to predict cluster
    predicted_clusters = model.predict(normalized_data)
    # print "\n\n### Predictions:"
    # print predicted_clusters

    predicted_clusters = pd.DataFrame(predicted_clusters, columns=["prediction_model"], index=normalized_data.index)
    predicted_clusters["prediction_euclidean"] = 0

    # drop target columns in centroids and in normalized_data so distance not based on them
    centroids.drop(labels=actual_cols, axis=1, inplace=True)
    normalized_data.drop(labels=actual_cols, axis=1, inplace=True)

    # get predictions based on euclidean distance
    for index, row in normalized_data.iterrows():
        distances = euclidean_distances(centroids, row.values.reshape(1, -1))
        cluster_index = np.array(distances).argmin()
        predicted_clusters.ix[index, 'prediction_euclidean'] = cluster_index


    # TODO: This also predicts always the same cluster for every row -> why?
    # find nearest centroid for each row of the given data
    print "\n\n### Predictions:\n"
    numeric_columns = clusters[0]._get_numeric_data().columns

    for tc in target_columns:

        print "\n\n\n\n##### Predict label:", tc
        correct_predict = 0
        correct_euclidean = 0
        abs_err_predict = 0
        abs_err_euclidean = 0

        for index, row in normalized_data.iterrows():
            print "\n#### Current row:", index
            cluster_index_euc = predicted_clusters["prediction_euclidean"].loc[index]
            print "### Euclidean Distance:", "Cluster found:", cluster_index_euc
            cluster_index_pred = predicted_clusters["prediction_model"].loc[index]
            print "### Model Prediction:", "Cluster found:", cluster_index_pred
            actual = unnormalized_data.loc[index][tc]
            print "## Actucal value:", actual
            print "## Cluster values:"
            if tc in numeric_columns:
                median = clusters[cluster_index_euc][tc].median()
                abs_err = abs(actual - median)
                abs_err_euclidean = abs_err_euclidean + abs_err
                print "# Euclidean Distance:", median, "Error:", abs_err
                median = clusters[cluster_index_pred][tc].median()
                abs_err = abs(actual - median)
                abs_err_predict = abs_err_predict + abs_err
                print "# Model Prediction:", median, "Error:", abs_err
            else:
                print "# Euclidean Distance:"
                print "Cluster shape:", clusters[cluster_index_euc][tc].shape[0]
                value_counts = clusters[cluster_index_euc][tc].value_counts()
                if len(value_counts) > 0:
                    majority = value_counts.idxmax(axis=1)
                else:
                    majority = np.NaN
                if majority == actual:
                    correct_euclidean = correct_euclidean + 1
                print "Majority voting:", majority
                print "# Model Prediction:"
                print "Cluster shape:", clusters[cluster_index_pred][tc].shape[0]
                value_counts = clusters[cluster_index_pred][tc].value_counts()
                if len(value_counts) > 0:
                    majority = value_counts.idxmax(axis=1)
                else:
                    majority = np.NaN
                if majority == actual:
                    correct_predict = correct_predict + 1
                print "Majority voting:", majority
        if tc in numeric_columns:
            print "\n\n#### Abs Error Euclidean:", abs_err_euclidean / float(normalized_data.shape[0])
            print "#### Abs Error Model:", abs_err_predict / float(normalized_data.shape[0])
        else:
            print "\n\n#### Correct Euclidean:", correct_euclidean, "in %:", float(correct_euclidean) / float(normalized_data.shape[0])
            print "#### Correct Model:", correct_predict, "in %:", float(correct_predict) / float(normalized_data.shape[0])
            print "#### Number of rows:", normalized_data.shape[0]



def test_clustering(file_name, method="Mean-Shift"):
    """ Test clustering for predictions (with test and train set)

    :param file_name: JSON file containing all data
    :type file_name: str
    :param method: Clustering Method to use: "Mean-Shift", "K-Means" or "DBSCAN"
    :type method: str
    """

    data_frame = prepare_data(file_name, budget_name="")
    df_train, df_test = train_test_split(data_frame, train_size=0.8)

    data_frame_original_test = get_overall_job_reviews(df_test.copy())

    if method == "Mean-Shift":
        model, clusters, centroids, min, max, vectorizers = do_clustering_mean_shift(df_train, find_best_params=False, explore_clusters=False)
    elif method == "K-Means":
        model, clusters, centroids, min, max, vectorizers = do_clustering_kmeans(df_train, find_best_params=False, explore_clusters=False)
    # TODO: With DBSCAN, clusters are based on density --> using centroids makes no sennse, have to cluster again!
    elif method == "DBSCAN":
        model, clusters, centroids, min, max, vectorizers = do_clustering_dbscan(df_train, find_best_params=False, explore_clusters=False)

    # # balance data set for experience_level or job_type
    # df_test = balance_data_set(df_test, "experience_level", relative_sampling=False)

    # remove rows without budget to predict budget
    df_test.ix[data_frame.budget == 0, 'budget'] = None
    df_test.dropna(subset=["budget"], how='any', inplace=True)

    # prepare test data
    df_test = prepare_test_data(df_test, centroids.columns, min, max, vectorizers=vectorizers, weighting=True)

    predict(model, data_frame_original_test, df_test, clusters, centroids, target_columns=['budget'])
    # predict(model, data_frame_original_test, df_test, clusters, centroids, target_columns=['job_type', 'experience_level'])
