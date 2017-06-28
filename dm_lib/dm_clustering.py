from dm_data_preparation import *
from dm_general import *
from sklearn.cluster import DBSCAN
from sklearn import metrics
import numpy as np
from dm_text_mining import addTextTokensToWholeDF
pd.set_option('display.max_columns', 200)

def prepare_data_clustering(data_frame, z_score_norm=False, add_text=False, weighting=True):
    """ Clean and prepare data specific to clustering

    :param data_frame: Pandas DataFrame that holds the data
    :type data_frame: pandas.DataFrame
    :return: Cleaned Pandas DataFrames once with only numerical attributes and once only text attributes
    :rtype: pandas.DataFrame
    """

    # declare total_charge as missing, if 0
    data_frame.ix[data_frame.total_charge == 0, 'total_charge'] = None

    # rows that don't contain total_charge
    data_frame.dropna(subset=["total_charge"], how='any', inplace=True)

    # overall feedbacks
    get_overall_job_reviews(data_frame)

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
        data_frame, token_names = addTextTokensToWholeDF(data_frame, text_data)

    # normalize
    # TODO store mean/std or min/max to normalize user job the same way
    if z_score_norm:
        data_frame = (data_frame - data_frame.mean()) / data_frame.std() # z-score
    else:
        data_frame = (data_frame-data_frame.min())/(data_frame.max()-data_frame.min()) # min max

    # # weighting
    # data_frame["client_feedback"] = data_frame["client_feedback"] * 10
    # data_frame["duration_weeks_total"] = data_frame["duration_weeks_total"] * 10
    # data_frame["duration_weeks_median"] = data_frame["duration_weeks_median"] * 5
    # data_frame["freelancer_count"] = data_frame["freelancer_count"] * 10
    # data_frame["total_charge"] = data_frame["total_charge"] * 20
    # data_frame["skills_number"] = data_frame["skills_number"] * 5
    # data_frame["feedback_for_client"] = data_frame["feedback_for_client"] * 20
    # data_frame["feedback_for_freelancer"] = data_frame["feedback_for_freelancer"] * 10
    # data_frame["job_type_Fixed"] = data_frame["job_type_Fixed"] * 20
    # data_frame["job_type_Hourly"] = data_frame["job_type_Hourly"] * 20

    if weighting:
        country_columns = [col for col in list(data_frame) if col.startswith('client_country')]
        data_frame[country_columns] = data_frame[country_columns] / len(country_columns)

        if add_text:
            data_frame[token_names] = data_frame[token_names] / len(token_names)

    # print data_frame, "\n"
    print_data_frame("After preparing for clustering", data_frame)
    # print_statistics(data_frame)

    return data_frame, text_data

def do_clustering_dbscan(file_name):
    """ Learn model for label 'budget' and return it

    :param file_name: JSON file containing all data
    :type file_name: str
    """

    find_best_params = False
    min_n_clusters = 10
    max_n_clusters = 500

    data_frame = prepare_data(file_name, budget_name="total_charge")
    data_frame_original = data_frame.copy()

    # prepare for clustering
    data_frame, text_data = prepare_data_clustering(data_frame)
    # print data_frame[0:5]

    if find_best_params:
        best_silhouette_score = -1000
        best_eps = -1
        best_min_samples = -1
        best_n_clusters = -1
        config_num = 1
        for eps in np.arange(0.5, 60, 0.1):
            for min_samples in range(2, 5, 1):
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
        db = DBSCAN(eps=1.0, min_samples=2).fit(data_frame)  # not weighted, min-max
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
            print "\n\n######################## \nPrint Clusters: \n########################"
            for cluster in clusters:
                print "Cluster: "+str(cluster["cluster_label"][0]), " --- Shape: ", cluster.shape
                # print '\033[94m', cluster[0:5], '\033[0m'
                print "########################\n"

        else:
            print "No good clustering"