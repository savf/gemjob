from dm_data_preparation import *
from dm_general import *
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
from dm_text_mining import addTextTokensToWholeDF
pd.set_option('display.max_columns', 200)

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
    data_frame['feedback_for_client'].fillna(data_frame['feedback_for_client'].mean(), inplace=True)
    data_frame['feedback_for_freelancer'].fillna(data_frame['feedback_for_freelancer'].mean(), inplace=True)
    data_frame['client_feedback'].fillna(data_frame['client_feedback'].mean(), inplace=True)

    # fill missing experience levels with random non-missing values
    filled_experience_levels = data_frame["experience_level"].dropna()
    data_frame["experience_level"] = data_frame.apply(
        lambda row: row["experience_level"] if row["experience_level"] is not None
        else random.choice(filled_experience_levels), axis=1)

    # convert everything to numeric
    data_frame = convert_to_numeric(data_frame, label_name="")

    # handle text
    data_frame, text_data = separate_text(data_frame)
    if add_text:
        data_frame = addTextTokensToWholeDF(data_frame, text_data)

    # normalize
    if z_score_norm:
        data_frame, _, _ = normalize_z_score(data_frame)
    else:
        data_frame, _, _ = normalize_min_max(data_frame)

    if weighting:
        data_frame = weight_data(data_frame)

    # print data_frame, "\n"
    # print_data_frame("After preparing for clustering", data_frame)
    # print_statistics(data_frame)

    return data_frame, text_data


def explore_clusters(clusters):
    """ Print stats and facts about the clusters

    :param clusters: List of clusters in the form of Pandas Data Frames
    :type clusters: list
    """
    print "\n\n\n#################### Explore clusters ####################\n"
    print "Number of clusters:", len(clusters)

    # TODO: store stats for each cluster into a file
    for cluster in clusters:
        print "\n\nCluster: " + str(cluster["cluster_label"][0]), " --- Shape: ", cluster.shape
        # print '\033[94m', cluster[0:5], '\033[0m'

        selected_numeric_colums = ['duration_weeks_median', 'duration_weeks_total', 'client_feedback', 'feedback_for_client', 'feedback_for_freelancer', 'total_charge', 'skills_number', 'snippet_length']
        for num_col in selected_numeric_colums:
            print num_col, " --- mean:", cluster[num_col].mean(), ", std:", cluster[num_col].std()


        selected_nominal_colums = ['client_country', 'experience_level', 'job_type', 'subcategory2']
        for nom_col in selected_nominal_colums:
            unique = cluster[nom_col].unique()
            val_counts = cluster[nom_col].value_counts()

            print nom_col, " --- unique values:", len(unique)
            # print '\033[94m\n', val_counts, '\033[0m\n'

    # TODO Overall analysis:
        # - average of unique values per nominal column
        # - average std deviation and std deviation of means for each numerical column
                # -> how different are means between clusters, how similar is data within cluster (std)
        # generate final score to easily compare all clustering algorithms and normalizations
    print "\n##########################################################\n"


def do_clustering_dbscan(file_name):
    """ Cluster using DBSCAN algorithm
    silhouette_score about 0.58 without removing columns
    silhouette_score about 0.64 WITH removing columns

    :param file_name: JSON file containing all data
    :type file_name: str
    """

    find_best_params = False
    min_n_clusters = 10
    max_n_clusters = 500

    data_frame = prepare_data(file_name, budget_name="total_charge")
    data_frame_original = get_overall_job_reviews(data_frame.copy())

    # prepare for clustering
    data_frame, text_data = prepare_data_clustering(data_frame, z_score_norm=False, add_text=True)
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
                unique_labels = set(labels)

                n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
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

        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        print "Number of clusters: ", n_clusters
        if n_clusters > min_n_clusters and n_clusters < max_n_clusters:
            silhouette_score = metrics.silhouette_score(data_frame, labels)
            print "Silhouette Coefficient: ", silhouette_score

            # cluster the original data frame
            data_frame["cluster_label"] = labels
            data_frame = data_frame[data_frame.cluster_label != -1] # remove noisy samples (have cluster label -1)

            data_frame_original = data_frame_original.join(data_frame["cluster_label"], how='inner')

            # print_data_frame("Clustered Data Frame", data_frame_original)

            gb = data_frame_original.groupby('cluster_label')
            clusters = [gb.get_group(x) for x in gb.groups]
            print "Number of clusters:", len(clusters)

            # TODO look at the clusters and see what we got ...
            # already difficult with one dataset -> how do we do this with hundreds of clusters?!
            explore_clusters(clusters)
        else:
            print "No good clustering"


def do_clustering_kmeans(file_name):
    """ Cluster using k-means algorithm
    silhouette_score about 0.54 (0.25 with z-score) without removing columns
    silhouette_score about 0.60 WITH removing columns

    :param file_name: JSON file containing all data
    :type file_name: str
    """

    find_best_params = False
    min_n_clusters = 10
    max_n_clusters = 500

    data_frame = prepare_data(file_name, budget_name="total_charge")
    data_frame_original = get_overall_job_reviews(data_frame.copy())

    # prepare for clustering
    data_frame, text_data = prepare_data_clustering(data_frame, z_score_norm=False, add_text=True)
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

        silhouette_score = metrics.silhouette_score(data_frame, labels)
        print "Silhouette Coefficient: ", silhouette_score

        # cluster the original data frame
        data_frame["cluster_label"] = labels
        data_frame = data_frame[data_frame.cluster_label != -1] # remove noisy samples (have cluster label -1)

        data_frame_original = data_frame_original.join(data_frame["cluster_label"], how='inner')

        # print_data_frame("Clustered Data Frame", data_frame_original)

        gb = data_frame_original.groupby('cluster_label')
        clusters = [gb.get_group(x) for x in gb.groups]
        print "Number of clusters:", len(clusters)

        # TODO look at the clusters and see what we got ...
        # already difficult with one dataset -> how do we do this with hundreds of clusters?!
        explore_clusters(clusters)