from sklearn import metrics
from sklearn.cluster import DBSCAN, KMeans, MeanShift
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split

from dm_data_preparation import *
from dm_general import *
from dm_text_mining import add_text_tokens_to_data_frame

ERROR_VALUE = -1

def prepare_data_clustering(data_frame, z_score_norm=False, add_text=False, weighting=True, do_log_transform=True):
    """ Clean and prepare data specific to clustering

    :param data_frame: Pandas DataFrame that holds the data
    :type data_frame: pandas.DataFrame
    :param z_score_norm: Use z-score normalization
    :type z_score_norm: bool
    :param add_text: Add text tokens
    :type add_text: bool
    :param weighting: Do weighting
    :type weighting: bool
    :param do_log_transform: Log transform highly skewed data
    :type do_log_transform: bool
    :return: Cleaned Pandas DataFrame with only numerical attributes and either min/max or mean/std, if text added also vectorizers
    :rtype: pandas.DataFrame
    """

    # drop columns that are unnecessary for clustering: are they?
        # idea: we don't want to predict_comparison anymore, we just want to cluster based on interesting attributes provided by user
    drop_unnecessary = ["date_created", "client_jobs_posted", "client_past_hires", "client_reviews_count"]
    data_frame.drop(labels=drop_unnecessary, axis=1, inplace=True)

    # declare total_charge as missing, if 0
    data_frame.ix[data_frame.total_charge == 0, 'total_charge'] = None

    # rows that don't contain total_charge
    data_frame.dropna(subset=["total_charge"], how='any', inplace=True)

    # declare feedbacks as missing, if 0
    data_frame.ix[data_frame.client_feedback == 0, 'client_feedback'] = None

    # remove rows with missing values

    # feedbacks
    data_frame.feedback_for_client.fillna(method='ffill', inplace=True)
    data_frame.feedback_for_freelancer.fillna(method='ffill', inplace=True)
    data_frame.client_feedback.fillna(method='ffill', inplace=True)

    # convert everything to numeric
    data_frame = convert_to_numeric(data_frame, label_name="")

    # handle text
    data_frame, text_data = separate_text(data_frame)
    if add_text:
        data_frame, vectorizers = add_text_tokens_to_data_frame(data_frame, text_data)
    else:
        vectorizers = {}

    # transform to log scale where skewed distribution
    treat_outliers_deletion(data_frame)
    if do_log_transform:
        data_frame = transform_log_scale(data_frame, add_to_df=False)

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


def prepare_test_data_clustering(data_frame, cluster_columns, min, max, vectorizers=None, weighting=True, do_log_transform=True):
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
    :param do_log_transform: Log transform highly skewed data
    :type do_log_transform: bool
    :return: Cleaned Pandas DataFrame with only numerical attributes
    :rtype: pandas.DataFrame
    """

    # drop columns that are unnecessary for clustering: are they?
        # idea: we don't want to predict_comparison anymore, we just want to cluster based on interesting attributes provided by user
    drop_unnecessary = ["date_created", "client_jobs_posted", "client_past_hires", "client_reviews_count"]
    data_frame.drop(labels=drop_unnecessary, axis=1, inplace=True)

    # declare total_charge as missing, if 0
    data_frame.ix[data_frame.total_charge == 0, 'total_charge'] = None

    # rows that don't contain total_charge
    data_frame.dropna(subset=["total_charge"], how='any', inplace=True)

    # declare feedbacks as missing, if 0
    data_frame.ix[data_frame.client_feedback == 0, 'client_feedback'] = None

    # remove rows with missing values

    # feedbacks
    data_frame.feedback_for_client.fillna(method='ffill', inplace=True)
    data_frame.feedback_for_freelancer.fillna(method='ffill', inplace=True)
    data_frame.client_feedback.fillna(method='ffill', inplace=True)

    # experience level
    # data_frame["experience_level"].dropna(how='any', inplace=True)

    # convert everything to numeric
    data_frame = convert_to_numeric(data_frame, label_name="")

    # handle text
    data_frame, text_data = separate_text(data_frame)
    if vectorizers is not None:
        data_frame, _ = add_text_tokens_to_data_frame(data_frame, text_data, vectorizers=vectorizers)

    # add missing columns (dummies, that were not in this data set)
    for col in cluster_columns:
        if col not in data_frame.columns:
            data_frame[col] = 0
    # remove columns not existing in clusters
    for col in data_frame.columns:
        if col not in cluster_columns:
            data_frame.drop(labels=[col], axis=1, inplace=True)

    # transform to log scale where skewed distribution
    if do_log_transform:
        data_frame = transform_log_scale(data_frame, add_to_df=False)

    # normalize
    data_frame, _, _ = normalize_min_max(data_frame, min, max)

    if weighting:
        data_frame = weight_data(data_frame)

    # order acording to cluster_columns (important!!! scikit does not look at labels!)
    # print data_frame[0:5], "\n"
    data_frame = data_frame.reindex_axis(cluster_columns, axis=1)
    # print data_frame[0:5], "\n"

    return data_frame

def prepare_single_job_clustering(data_frame, cluster_columns, min, max, vectorizers=None, weighting=True, do_log_transform=True):
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
    :param do_log_transform: Log transform highly skewed data
    :type do_log_transform: bool
    :return: Cleaned Pandas DataFrame with only numerical attributes
    :rtype: pandas.DataFrame
    """

    # declare total_charge as missing, if 0
    data_frame.ix[data_frame.total_charge == 0, 'total_charge'] = None

    # declare feedbacks as missing, if 0
    data_frame.ix[data_frame.client_feedback == 0, 'client_feedback'] = None

    # identify target columns and drop them
    target_columns = data_frame.columns[data_frame.isnull().any()].tolist()
    data_frame.drop(labels=target_columns, axis=1, inplace=True)

    # don't try to predict text
    text_col_names = list(set(["skills", "snippet", "title"]).intersection(set(target_columns)))
    for c in text_col_names:
        target_columns.remove(c)

    # remove unnecessary columns
    drop_unnecessary = ["date_created", "client_jobs_posted", "client_past_hires", "client_reviews_count"]
    for c in drop_unnecessary:
        if c in data_frame.columns:
            data_frame.drop(labels=[c], axis=1, inplace=True)

    # convert everything to numeric
    data_frame = convert_to_numeric(data_frame, label_name="")

    # handle text
    data_frame, text_data = separate_text(data_frame)
    if vectorizers is not None:
        data_frame, _ = add_text_tokens_to_data_frame(data_frame, text_data, vectorizers=vectorizers)

    # add missing columns (dummies, that were not in this data set)
    for col in cluster_columns:
        if col not in data_frame.columns:
            data_frame[col] = 0
    # remove columns not existing in clusters
    for col in data_frame.columns:
        if col not in cluster_columns:
            data_frame.drop(labels=[col], axis=1, inplace=True)

    # print_data_frame("after adding all cols", data_frame)

    # transform to log scale where skewed distribution
    if do_log_transform:
        data_frame = transform_log_scale(data_frame, add_to_df=False)

    # normalize
    data_frame, _, _ = normalize_min_max(data_frame, min, max)

    if weighting:
        data_frame = weight_data(data_frame)

    # order acording to cluster_columns (important!!! scikit does not look at labels!)
    # print data_frame[0:5], "\n"
    data_frame = data_frame.reindex_axis(cluster_columns, axis=1)
    # print data_frame[0:5], "\n"

    return data_frame, target_columns

def explore_clusters(clusters, original_data_frame, silhouette_score, name=""):
    """ Print stats and facts about the clusters

    :param clusters: Dict of clusters in the form of Pandas Data Frames (key is cluster number, does not necessarily start with 0!)
    :type clusters: dict
    :return: Final score for clusters (lower is better)
    :rtype: float
    """
    selected_nominal_colums = ['client_country', 'experience_level', 'job_type', 'subcategory2']
    selected_numeric_colums = ['duration_weeks_median', 'client_feedback', 'budget',
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
    for cluster in clusters.itervalues():
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


def do_clustering_dbscan(data_frame, find_best_params=False, do_explore=True, min_rows_per_cluster=3, do_log_transform=True):
    """ Cluster using DBSCAN algorithm
    silhouette_score about 0.58 without removing columns
    silhouette_score about 0.64 WITH removing columns

    :param data_frame: Pandas DataFrame holding non-normalized data
    :type data_frame: pandas.DataFrame
    :param find_best_params: Find best parameters for clustering
    :type find_best_params: bool
    :param do_explore: Print stats about clusters
    :type do_explore: bool
    :param min_rows_per_cluster: Minimum number of rows a cluster should have
    :type min_rows_per_cluster: int
    :param do_log_transform: Log transform highly skewed data
    :type do_log_transform: bool
    :return: model, clusters based on unnormalized data, centroids (normalized), min, max, vectorizers
    :rtype: multiple
    """

    min_n_clusters = 10
    max_n_clusters = 500
    # eps = 38.5 # weighted, z-score
    # min_samples = 2 # weighted, z-score
    eps = 1.0 # no text, min-max
    min_samples = 2 # no text, min-max

    data_frame_original = data_frame.copy()

    # prepare for clustering
    data_frame, min, max, vectorizers = prepare_data_clustering(data_frame, z_score_norm=False, add_text=True, do_log_transform=do_log_transform)
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
        eps = best_eps
        min_samples = best_min_samples

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data_frame)
    labels = db.labels_
    # centroids = pd.DataFrame(db.cluster_centers_, columns=data_frame.columns)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print "Number of clusters: ", n_clusters
    silhouette_score = metrics.silhouette_score(data_frame, labels)

    # cluster the original data frame
    data_frame["cluster_label"] = labels
    data_frame = data_frame[data_frame.cluster_label != -1] # remove noisy samples (have cluster label -1)

    if min_rows_per_cluster > 1:
        value_counts = data_frame["cluster_label"].value_counts()
        small_clusters = value_counts.loc[value_counts < min_rows_per_cluster]
        # centroids.drop(small_clusters.index, inplace=True)
        data_frame = data_frame[-data_frame["cluster_label"].isin(list(small_clusters.index))]
        # #print
        # value_counts = data_frame["cluster_label"].value_counts()
        # print "\n### Remaining indices: ", value_counts

    data_frame_original = data_frame_original.join(data_frame["cluster_label"], how='inner')

    # declare budget as missing, if hourly job
    data_frame_original.ix[data_frame_original.job_type == 'Hourly', 'budget'] = None

    gb = data_frame_original.groupby('cluster_label')
    # clusters = [gb.get_group(x) for x in gb.groups]
    clusters = {gb.get_group(x)['cluster_label'][0]: gb.get_group(x) for x in gb.groups}
    # print  "\n### Cluster Keys: ", clusters.keys()

    if do_explore:
        explore_clusters(clusters, data_frame_original, silhouette_score, "DBSCAN")

    return db, clusters, None, min, max, vectorizers


def do_clustering_kmeans(data_frame, find_best_params=False, do_explore=True, min_rows_per_cluster=3, do_log_transform=True):
    """ Cluster using k-means algorithm
    silhouette_score about 0.54 (0.25 with z-score) without removing columns
    silhouette_score about 0.60 WITH removing columns

    :param data_frame: Pandas DataFrame holding non-normalized data
    :type data_frame: pandas.DataFrame
    :param find_best_params: Find best parameters for clustering
    :type find_best_params: bool
    :param do_explore: Print stats about clusters
    :type do_explore: bool
    :param min_rows_per_cluster: Minimum number of rows a cluster should have
    :type min_rows_per_cluster: int
    :param do_log_transform: Log transform highly skewed data
    :type do_log_transform: bool
    :return: model, clusters based on unnormalized data, centroids (normalized), min, max, vectorizers
    :rtype: multiple
    """

    min_n_clusters = 10
    max_n_clusters = 500
    # n_clusters = 98  # without text, min-max scaling
    n_clusters = 65 # with text, min-max scaling

    data_frame_original = data_frame.copy()

    # prepare for clustering
    data_frame, min, max, vectorizers = prepare_data_clustering(data_frame, z_score_norm=False, add_text=True, do_log_transform=do_log_transform)
    # print data_frame[0:5]

    if find_best_params:
        best_silhouette_score = -1000
        best_k = -1
        config_num = 1

        for k in range(min_n_clusters, max_n_clusters, 5): # ran it until 250, began to get worse after 98
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
        n_clusters = best_k

    kmeans = KMeans(n_clusters=n_clusters).fit(data_frame)
    labels = kmeans.labels_
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=data_frame.columns)

    silhouette_score = metrics.silhouette_score(data_frame, labels)

    # cluster the original data frame
    data_frame["cluster_label"] = labels
    data_frame = data_frame[data_frame.cluster_label != -1] # remove noisy samples (have cluster label -1)

    if min_rows_per_cluster > 1:
        value_counts = data_frame["cluster_label"].value_counts()
        small_clusters = value_counts.loc[value_counts < min_rows_per_cluster]
        centroids.drop(small_clusters.index, inplace=True)
        data_frame = data_frame[-data_frame["cluster_label"].isin(list(small_clusters.index))]
        # #print
        # value_counts = data_frame["cluster_label"].value_counts()
        # print "\n### Remaining indices: ", value_counts

    data_frame_original = data_frame_original.join(data_frame["cluster_label"], how='inner')

    # declare budget as missing, if hourly job
    data_frame_original.ix[data_frame_original.job_type == 'Hourly', 'budget'] = None

    gb = data_frame_original.groupby('cluster_label')
    # clusters = [gb.get_group(x) for x in gb.groups]
    clusters = {gb.get_group(x)['cluster_label'][0]: gb.get_group(x) for x in gb.groups}
    # print  "\n### Cluster Keys: ", clusters.keys()

    if do_explore:
        explore_clusters(clusters, data_frame_original, silhouette_score, "K-Means")

    return kmeans, clusters, centroids, min, max, vectorizers


def do_clustering_mean_shift(data_frame, find_best_params=False, do_explore=True, min_rows_per_cluster=3, do_log_transform=True):
    """ Cluster using mean-shift algorithm
    silhouette_score about 0.58 without removing columns
    silhouette_score about 0.65 WITH removing columns
    silhouette_score about 0.85 with removing columns WITH z-score BUT only 2 clusters

    :param data_frame: Pandas DataFrame holding non-normalized data
    :type data_frame: pandas.DataFrame
    :param find_best_params: Find best parameters for clustering
    :type find_best_params: bool
    :param do_explore: Print stats about clusters
    :type do_explore: bool
    :param min_rows_per_cluster: Minimum number of rows a cluster should have
    :type min_rows_per_cluster: int
    :param do_log_transform: Log transform highly skewed data
    :type do_log_transform: bool
    :return: model, clusters based on unnormalized data, centroids (normalized), min, max, vectorizers
    :rtype: multiple
    """

    bandwidth = 1.2

    data_frame_original = data_frame.copy()

    # prepare for clustering
    data_frame, min, max, vectorizers = prepare_data_clustering(data_frame, z_score_norm=False, add_text=True, do_log_transform=do_log_transform)

    # print data_frame[0:5]

    if find_best_params:
        best_silhouette_score = -1000
        best_bandwidth = -1
        config_num = 1

        for bandwidth in np.arange(0.5, 2.0, 0.1):
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
        bandwidth = best_bandwidth

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(data_frame) # min-max
    # ms = MeanShift(bandwidth=38.0, bin_seeding=True).fit(data_frame) # z-score
    labels = ms.labels_
    centroids = pd.DataFrame(ms.cluster_centers_, columns=data_frame.columns)

    silhouette_score = metrics.silhouette_score(data_frame, labels)
    print "silhouette_score =",silhouette_score

    # cluster the original data frame
    data_frame["cluster_label"] = labels
    data_frame = data_frame[data_frame.cluster_label != -1] # remove noisy samples (have cluster label -1)

    if min_rows_per_cluster > 1:
        value_counts = data_frame["cluster_label"].value_counts()
        small_clusters = value_counts.loc[value_counts < min_rows_per_cluster]
        # print "\n### All indices: ", centroids.index
        centroids.drop(small_clusters.index, inplace=True)
        # print "\n### Remaining indices: ", centroids.index
        data_frame = data_frame[-data_frame["cluster_label"].isin(list(small_clusters.index))]
        # #print
        # value_counts = data_frame["cluster_label"].value_counts()
        # print "\n### Remaining indices: ", value_counts

    data_frame_original = data_frame_original.join(data_frame["cluster_label"], how='inner')

    # declare budget as missing, if hourly job
    data_frame_original.ix[data_frame_original.job_type == 'Hourly', 'budget'] = None

    gb = data_frame_original.groupby('cluster_label')
    # clusters = [gb.get_group(x) for x in gb.groups]
    clusters = {gb.get_group(x)['cluster_label'][0]: gb.get_group(x) for x in gb.groups}
    # print  "\n### Cluster Keys: ", clusters.keys()

    if do_explore:
        explore_clusters(clusters, data_frame_original, silhouette_score, "Mean-Shift")

    return ms, clusters, centroids, min, max, vectorizers


def predict(unnormalized_data, normalized_data, clusters, centroids, target_columns, do_reweighting=True):
    """ Predict columns based on distance to cluster centroids

    :param unnormalized_data: Pandas DataFrames holding non-normalized data
    :type unnormalized_data: pandas.DataFrame
    :param normalized_data: Pandas DataFrames holding normalized data
    :type normalized_data: pandas.DataFrame
    :param clusters: Dict of clusters in the form of Pandas Data Frames (key is cluster number, does not necessarily start with 0!)
    :type clusters: dict
    :param centroids: minimum values
    :type centroids: pandas.DataFrame
    :param target_columns: columns to predict_comparison
    :type target_columns: list
    :return: Pandas Dataframe with predictions for each row of the input data and the "cluster_size" as its own column
    :rtype: pandas.DataFrame
    :param do_reweighting: Remove tokens not in test example and reweight tokens
    :type do_reweighting: bool
    """

    # get actual target_columns (dummies)
    actual_cols_normalized = []
    for tc in target_columns:
        actual_cols_normalized = actual_cols_normalized + [col for col in list(normalized_data) if col.startswith(tc)]


    # drop target columns in centroids and in normalized_data so distance not based on them
    centroids.drop(labels=actual_cols_normalized, axis=1, inplace=True)
    normalized_data.drop(labels=actual_cols_normalized, axis=1, inplace=True)

    # get predictions based on euclidean distance
    predicted_clusters = pd.DataFrame(0, columns=["prediction_euclidean"], index=normalized_data.index)


    # print "\nCentroid indices", centroids.index
    for index, row in normalized_data.iterrows():
        row_df = pd.DataFrame(row.values.reshape(1, -1), index=[0], columns=list(normalized_data.columns))
        if do_reweighting:
            row_rw, centroids_rw = reduce_tokens_to_single_job(row_df, centroids.copy())
        else:
            row_rw = row_df
            centroids_rw = centroids
        distances = euclidean_distances(centroids_rw, row_rw)
        cluster_index = np.array(distances).argmin()
        cluster_index = centroids.index[cluster_index]
        predicted_clusters.ix[index, 'prediction_euclidean'] = cluster_index

    # find nearest centroid for each row of the given data
    all_columns = list(clusters.itervalues().next().columns)
    all_columns = [x for x in all_columns if x not in ["cluster_label", "skills", "title", "snippet", "client_country", "date_created", "client_reviews_count", ]]
    numeric_columns = list(clusters.itervalues().next()._get_numeric_data().columns)
    numeric_columns.remove("experience_level")

    unnormalized_data = unnormalized_data[all_columns]

    # add stats columns
    for tc in all_columns:
        # if tc not in unnormalized_data.index and tc in numeric_columns:
        #     unnormalized_data[tc] = 0
        # elif tc not in unnormalized_data.index :
        #     unnormalized_data[tc] = ""

        if tc in numeric_columns:
            unnormalized_data[tc + "_prediction"] = 0
            unnormalized_data[tc + "_mean"] = 0
            unnormalized_data[tc + "_min"] = 0
            unnormalized_data[tc + "_max"] = 0
            unnormalized_data[tc + "_std"] = 0
            unnormalized_data[tc + "_25quantile"] = 0
            unnormalized_data[tc + "_75quantile"] = 0
        else:
            unnormalized_data[tc + "_prediction"] = ""
            unnormalized_data[tc + "_value_counts"] = ""

    unnormalized_data["cluster_size"] = 0

    for index, _ in normalized_data.iterrows():
        for tc in all_columns:
            # print "\n\n\n\n##### Predict label:", tc
            # print "\n#### Current row:", index
            cluster_index_euc = predicted_clusters["prediction_euclidean"].loc[index]
            # print "Cluster found:", cluster_index_euc
            # print "Cluster shape:", clusters[cluster_index_euc].shape
            unnormalized_data.set_value(index, "cluster_size", clusters[cluster_index_euc].shape[0])

            actual = unnormalized_data.loc[index][tc]
            # print "## Actucal value:", actual

            if tc in numeric_columns:
                median = clusters[cluster_index_euc][tc].median()
                # if np.isnan(median):
                #     print "#ERROR for "+tc+": no median found!"
                #     # unnormalized_data.set_value(index, tc, ERROR_VALUE)
                # else:
                # print "# Prediction:", median #, "Error:", abs_err
                if not np.isnan(median):
                    unnormalized_data.set_value(index, (tc + "_prediction"), median)
                    unnormalized_data.set_value(index, (tc + "_mean"), clusters[cluster_index_euc][tc].mean())
                    unnormalized_data.set_value(index, (tc + "_min"), clusters[cluster_index_euc][tc].min())
                    unnormalized_data.set_value(index, (tc + "_max"), clusters[cluster_index_euc][tc].max())
                    unnormalized_data.set_value(index, (tc + "_std"), clusters[cluster_index_euc][tc].std())
                    unnormalized_data.set_value(index, (tc + "_25quantile"), clusters[cluster_index_euc][tc].quantile(.25))
                    unnormalized_data.set_value(index, (tc + "_75quantile"), clusters[cluster_index_euc][tc].quantile(.75))
            else:
                value_counts = clusters[cluster_index_euc][tc].value_counts()
                if len(value_counts) > 0:
                    majority = value_counts.idxmax(axis=1)
                else:
                    # print "#ERROR: no majority found!"
                    majority = np.NaN
                # print "Majority voting:", majority
                unnormalized_data.set_value(index, (tc + "_prediction"), majority)
                value_counts_string = ""
                if tc == "experience_level":
                    experience_level_names = ["Entry Level", "Intermediate", "Expert"]
                    for k, v in value_counts.iteritems():
                        value_counts_string = value_counts_string + experience_level_names[int(k) - 1] + ": <span style='float: right;'>" + str(v) + "</span><br>"
                else:
                    for k, v in value_counts.iteritems():
                        value_counts_string = value_counts_string + str(k) + ": <span style='float: right;'>" + str(v) + "</span><br>"
                unnormalized_data.set_value(index, (tc + "_value_counts"), value_counts_string)

    return unnormalized_data



def predict_comparison(model, unnormalized_data, normalized_data, clusters, centroids, target_columns, do_reweighting=False):
    """ Prints stats to compare prediction based on model and on eucledean distance to centroids

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
    :param target_columns: columns to predict_comparison
    :type target_columns: list
    :param do_reweighting: Remove tokens not in test example and reweight tokens
    :type do_reweighting: bool
    """

    # get actual target_columns (dummies)
    actual_cols = []
    for tc in target_columns:
        actual_cols = actual_cols + [col for col in list(normalized_data) if col.startswith(tc)]

    # set columns to predict_comparison to 0
    normalized_data[actual_cols] = 0
    # use model to predict_comparison cluster
    predicted_clusters = model.predict(normalized_data)
    # print "\n\n### Predictions:"
    # print predicted_clusters
    numb_in_outlier_cluster = 0
    for i in range(0, len(predicted_clusters)):
        try:
            predicted_clusters[i] = centroids.index[predicted_clusters[i]]
        except:
            predicted_clusters[i] = centroids.index[0]
            numb_in_outlier_cluster = numb_in_outlier_cluster+1

    # print "\n\n### Predictions:"
    # print predicted_clusters

    predicted_clusters = pd.DataFrame(predicted_clusters, columns=["prediction_model"], index=normalized_data.index)
    predicted_clusters["prediction_euclidean"] = 0
    predicted_clusters["prediction_euclidean_reweighted"] = 0

    # drop target columns in centroids and in normalized_data so distance not based on them
    centroids.drop(labels=actual_cols, axis=1, inplace=True)
    normalized_data.drop(labels=actual_cols, axis=1, inplace=True)

    # get predictions based on euclidean distance
    for index, row in normalized_data.iterrows():
        row_df = pd.DataFrame(row.values.reshape(1, -1), index=[0], columns=list(normalized_data.columns))
        # print row_df
        distances = euclidean_distances(centroids, row_df)
        cluster_index = np.array(distances).argmin()
        cluster_index = centroids.index[cluster_index]
        predicted_clusters.ix[index, 'prediction_euclidean'] = cluster_index

        if do_reweighting:
            # reweighted
            row_rw, centroids_rw = reduce_tokens_to_single_job(row_df.copy(), centroids.copy())
            distances = euclidean_distances(centroids_rw, row_rw)
            cluster_index = np.array(distances).argmin()
            cluster_index = centroids.index[cluster_index]
            predicted_clusters.ix[index, 'prediction_euclidean_reweighted'] = cluster_index


    # find nearest centroid for each row of the given data
    numeric_columns = list(clusters.itervalues().next()._get_numeric_data().columns)
    numeric_columns.remove("experience_level")

    for tc in target_columns:

        print "\n\n\n\n##### Predict label:", tc
        correct_predict = 0
        correct_euclidean = 0
        correct_euclidean_rw = 0
        abs_err_predict = 0
        abs_err_euclidean = 0
        abs_err_euclidean_rw = 0
        errors_euc = 0
        errors_euc_rw = 0
        errors_mod = 0

        if tc in numeric_columns:
            unnormalized_data[tc+"_prediction"] = 0
        else:
            unnormalized_data[tc + "_prediction"] = ""

        for index, row in normalized_data.iterrows():
            print "\n#### Current row:", index

            cluster_index_euc = predicted_clusters["prediction_euclidean"].loc[index]
            print "### Euclidean Distance:", "Cluster found:", cluster_index_euc

            if do_reweighting:
                cluster_index_euc_rw = predicted_clusters["prediction_euclidean_reweighted"].loc[index]
                print "### Euclidean Distance:", "Cluster found:", cluster_index_euc_rw

            cluster_index_pred = predicted_clusters["prediction_model"].loc[index]
            print "### Model Prediction:", "Cluster found:", cluster_index_pred

            actual = unnormalized_data.loc[index][tc]
            print "## Actucal value:", actual

            print "## Cluster values:"
            if tc in numeric_columns:

                median = clusters[cluster_index_euc][tc].median()
                if np.isnan(median):
                    print "#ERROR: no median found!"
                    errors_euc = errors_euc+1
                    unnormalized_data.set_value(index, tc + "_prediction", 0)
                else :
                    abs_err = abs(actual - median)
                    abs_err_euclidean = abs_err_euclidean + abs_err
                    print "# Euclidean Distance:", median, "Error:", abs_err
                    unnormalized_data.set_value(index, tc + "_prediction", median)

                if do_reweighting:
                    median = clusters[cluster_index_euc_rw][tc].median()
                    if np.isnan(median):
                        print "#ERROR: no median found!"
                        errors_euc_rw = errors_euc_rw + 1
                        unnormalized_data.set_value(index, tc + "_prediction", 0)
                    else:
                        abs_err = abs(actual - median)
                        abs_err_euclidean_rw = abs_err_euclidean_rw + abs_err
                        print "# Euclidean Distance Reweighted:", median, "Error:", abs_err
                        unnormalized_data.set_value(index, tc + "_prediction", median)

                median = clusters[cluster_index_pred][tc].median()
                if np.isnan(median):
                    print "#ERROR: no median found!"
                    errors_mod = errors_mod + 1
                else:
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
                unnormalized_data.set_value(index, tc + "_prediction", majority)

                if do_reweighting:
                    print "# Euclidean Distance Reweighted:"
                    print "Cluster shape:", clusters[cluster_index_euc_rw][tc].shape[0]
                    value_counts = clusters[cluster_index_euc_rw][tc].value_counts()
                    if len(value_counts) > 0:
                        majority = value_counts.idxmax(axis=1)
                    else:
                        majority = np.NaN
                    if majority == actual:
                        correct_euclidean_rw = correct_euclidean_rw + 1
                    print "Majority voting:", majority
                    unnormalized_data.set_value(index, tc + "_prediction", majority)

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
            print "#### Number of NaN Errors Euclidean:", errors_euc
            if do_reweighting:
                print "#### Abs Error Euclidean Reweighted:", abs_err_euclidean_rw / float(normalized_data.shape[0])
                print "#### Number of NaN Errors Euclidean Reweighted:", errors_euc_rw

            print "#### Abs Error Model:", abs_err_predict / float(normalized_data.shape[0])
            print "#### Number of NaN Errors Model:", errors_mod
        else:
            print "\n\n#### Correct Euclidean:", correct_euclidean, "in %:", float(correct_euclidean) / float(normalized_data.shape[0])
            if do_reweighting:
                print "#### Correct Euclidean Reweighted:", correct_euclidean_rw, "in %:", float(correct_euclidean_rw) / float(normalized_data.shape[0])
            print "#### Correct Model:", correct_predict, "in %:", float(correct_predict) / float(normalized_data.shape[0])
        print "#### Number of rows:", normalized_data.shape[0]
        print "#### Number of rows predicted to be in outlier cluster (removed from clusters):", numb_in_outlier_cluster

        return unnormalized_data.loc[normalized_data.index]


def test_clustering(file_name, method="Mean-Shift", target="budget"):
    """ Test clustering for predictions (with test and train set)

    :param file_name: JSON file containing all data
    :type file_name: str
    :param method: Clustering Method to use: "Mean-Shift", "K-Means" or "DBSCAN"
    :type method: str
    :param target: Target label to predict
    :type target: str
    """

    target_columns = [target]

    data_frame = prepare_data(file_name)
    # print_data_frame("After preparing data", data_frame)
    df_train, df_test = train_test_split(data_frame, train_size=0.8)

    data_frame_original_test = df_test.copy()

    if method == "Mean-Shift":
        # model, clusters, centroids, min, max, vectorizers = do_clustering_mean_shift(df_train.copy(), find_best_params=False, do_explore=False, do_log_transform=False)
        model_log, clusters_log, centroids_log, min_log, max_log, vectorizers_log = do_clustering_mean_shift(df_train.copy(), find_best_params=False, do_explore=False, do_log_transform=True)
    elif method == "K-Means":
        # model, clusters, centroids, min, max, vectorizers = do_clustering_kmeans(df_train.copy(), find_best_params=False, do_explore=False)
        model_log, clusters_log, centroids_log, min_log, max_log, vectorizers_log = do_clustering_kmeans(df_train.copy(), find_best_params=False, do_explore=False, do_log_transform=True)
    elif method == "DBSCAN":
        model, clusters, centroids, min, max, vectorizers = do_clustering_dbscan(df_train, find_best_params=True, do_explore=True, do_log_transform=True)
        return # With DBSCAN, clusters are based on density --> using centroids makes no sense and is also not possible.

    if target == "budget":
        # remove rows without budget to predict_comparison budget
        df_test.ix[df_test.job_type == 'Hourly', 'budget'] = None
        df_test.dropna(subset=["budget"], how='any', inplace=True)
    elif target == "job_type" or target == "experience_level" or target == "subcategory2":
        df_test = balance_data_set(df_test, target, relative_sampling=False)
    if target == "job_type":
        target_columns = [target, "budget", "workload"] # remove attributes that give away the job type
    if target == "feedback_for_client":
        df_test.dropna(subset=["feedback_for_client"], how='any', inplace=True)
        target_columns = [target, "feedback_for_freelancer", "client_feedback"]  # remove attributes that give away the feedback

    target_columns.append("client_payment_verification_status")  # not available for user job
    target_columns.append("total_charge")  # not available for user job

    # prepare test data
    df_test_log = prepare_test_data_clustering(df_test.copy(), centroids_log.columns, min_log, max_log, vectorizers=vectorizers_log, weighting=True, do_log_transform=True)
    # df_test = prepare_test_data_clustering(df_test.copy(), centroids.columns, min, max, vectorizers=vectorizers, weighting=True, do_log_transform=False)

    # predict(data_frame_original_test, df_test.drop(df_test.index[1:-1]), clusters, centroids, target_columns=['experience_level'])
    # unnormalized_data = predict_comparison(model, data_frame_original_test, df_test, clusters, centroids, target_columns=['budget'], do_reweighting=False)
    # unnormalized_data = predict_comparison(model, data_frame_original_test.copy(), df_test, clusters, centroids, target_columns=[target], do_reweighting=False)
    # unnormalized_data_log = predict_comparison(model_log, data_frame_original_test.copy(), df_test_log, clusters_log, centroids_log, target_columns=[target], do_reweighting=False)
    unnormalized_data_log = predict(data_frame_original_test.loc[df_test_log.index], df_test_log, clusters_log, centroids_log, target_columns=target_columns, do_reweighting=False)
    # predict_comparison(model, data_frame_original_test, df_test, clusters, centroids, target_columns=['subcategory2'], do_reweighting=True)
    # predict_comparison(model, data_frame_original_test, df_test, clusters, centroids, target_columns=['client_feedback'])

    print "\n"
    if target in ["budget", "client_feedback", "feedback_for_client", "feedback_for_freelancer"]:
        # evaluate_regression(unnormalized_data[target], unnormalized_data[target + '_prediction'], target)
        # print "\nLog transfomed:"
        # return evaluate_regression(unnormalized_data_log[target], unnormalized_data_log[target + '_prediction'], target)
        return evaluate_regression(unnormalized_data_log[target], unnormalized_data_log[target + '_prediction'], target)
    elif target == "job_type" or target == "experience_level" or target == "subcategory2":
        # evaluate_classification(unnormalized_data[target], unnormalized_data[target + '_prediction'], target)
        # print "\nLog transfomed:"
        return evaluate_classification(unnormalized_data_log[target], unnormalized_data_log[target + '_prediction'], target)
