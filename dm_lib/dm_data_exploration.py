import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats.mstats import normaltest

from dm_data_preparation import *
from dm_general import print_statistics, print_correlations


def perc_convert(ser):
  return ser/float(ser[-1])


def same_mean(series_1, series_2, significance):
    """ Check the variance and distribution and then make hypothesis test for the mean

    The variance and normal distribution test is needed to check whether a t-test could
    be used, since these two requirements are needed for the t-test. If these requirements
    are not met, the Mann-Whitney-Wilcoxon RankSum test is used

    :type series_1: pandas.Series
    :type series_2: pandas.Series
    :param significance: The significance is normally 5%
    """
    normaltest_series_1 = normaltest(series_1)
    normaltest_series_2 = normaltest(series_2)
    if series_1.var() == series_2.var() and\
        normaltest_series_1[1] <= significance and\
            normaltest_series_2[1] <= significance:
        result, p_value = stats.ttest_ind(series_1, series_2)
    else:
        result, p_value = stats.ranksums(series_1, series_2)
    # A small p value means the probability that values like the ones occur given that
    # both series have the same mean is small -> They don't have the same mean
    if p_value <= significance:
        return False
    else:
        return True


def explore_data(file_name,budget_name="total_charge"):
    """ Print some stats and plot some stuff

    :param file_name: JSON file containing all data
    :type file_name: str
    :param budget_name: Use either "budget" or "total_charge"
    :type file_name: str
    """
    data_frame = prepare_data(file_name, budget_name=budget_name)

    no_missing = data_frame.dropna(subset=['client_payment_verification_status'])
    only_missing = data_frame.loc[data_frame['client_payment_verification_status'].isnull()]

    client_feedback_mean_missing = same_mean(no_missing['client_feedback'], only_missing['client_feedback'], 0.05)
    client_feedback_mean_verified = same_mean(data_frame.loc[data_frame['client_payment_verification_status'] ==
                                                             'VERIFIED', 'client_feedback'],
                                              data_frame.loc[data_frame['client_payment_verification_status'] !=
                                                             'VERIFIED', 'client_feedback'], 0.05)
    print "Mean of client feedback for missing payment verification and non missing are the same: {}"\
        .format(client_feedback_mean_missing)
    print "Mean of client feedback for payment verification status VERIFIED or not are the same: {}"\
        .format(client_feedback_mean_verified)

    print "## crosstab job_type vs experience_level"
    print pd.crosstab(data_frame["job_type"], data_frame["experience_level"], margins=True).apply(perc_convert, axis=1)

    drop_unnecessary = ["client_country"]
    data_frame.drop(labels=drop_unnecessary, axis=1, inplace=True)

    data_frame = convert_to_numeric(data_frame, "")

    treat_outliers_deletion(data_frame)
    treat_outliers_log_scale(data_frame)

    for attr in ["client_feedback", "timestamp", "freelancer_count", "workload", "total_hours"]:
        # data_frame[attr].plot(kind='hist', legend=True, title=attr)
        # data_frame.hist(column=attr, bins=30)
        chart, ax = plt.subplots()
        # sns.distplot(data_frame[attr], hist_kws={'cumulative': True}, kde_kws={'cumulative': True}, ax=ax)
        #stats.probplot(data_frame[attr], plot=ax)
        #data_frame.boxplot(column=attr, ax=ax)
        #data_frame.plot.scatter(x='total_charge', y=attr, logx=True)
        sns.regplot(x=data_frame['freelancer_count'],
                    y=data_frame[attr], ax=ax, order=0, lowess=True)
        #sns.residplot(x=data_frame['total_charge'], y=data_frame[attr], ax=ax, order=0, lowess=True)
        ax.set_title("LOWESS for " + attr)
        #plt.show()
    print_statistics(data_frame)
    print_correlations(data_frame, store=True)
