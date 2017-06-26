import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats.mstats import normaltest
import numpy as np
from dm_data_preparation import *
from pandas.core.dtypes.dtypes import CategoricalDtype
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


def prepare_data_raw(file_name):
    """ Similar function to prepare_data in dm_data_preparation but without any data modification

    :param file_name: File name where data is stored
    :type file_name: str
    :return: DataFrame with raw data
    :rtype: pandas.DataFrame
    """
    data_frame = create_data_frame(file_name)
    data_frame.columns = [c.replace('.', '_') for c in
                          data_frame.columns]  # so we can access a column with "data_frame.client_reviews_count"
    print_data_frame("Before changing data", data_frame)

    # set id
    data_frame.set_index("id", inplace=True)

    # convert total_charge and freelancer_count to number
    data_frame["total_charge"] = pd.to_numeric(data_frame["total_charge"])
    data_frame["freelancer_count"] = pd.to_numeric(data_frame["freelancer_count"])

    # convert experience level from numeric to categorical
    experience_levels = ['beginner', 'intermediate', 'expert']
    data_frame['experience_level'] = pd.cut(data_frame['experience_level'], len(experience_levels),
                                            labels=experience_levels)

    return data_frame


def get_datatype_safely(dtype):
    if isinstance(dtype, CategoricalDtype):
        return np.object_
    return dtype


def plot_value_distributions(data_series, logx=False, logy=False):
    """ Plot a histogram and a KDE for the given Series

    :param data_series: The data series to be plotted
    :type data_series: pandas.Series
    :param logx: Plot the x axis in log scale
    :type logx: bool
    :param logy: Plot the y axis in log scale
    :type logy: bool
    """
    # Check if data is numeric or not
    fig, axarr = plt.subplots(ncols=2)

    if np.issubdtype(get_datatype_safely(data_series.dtype), np.number):
        data_series.hist(bins=30, ax=axarr[0])
        data_series.plot.density(ax=axarr[1])
    else:
        data_series.value_counts().plot(kind='bar', ax=axarr[0])
        number_of_categories = len(data_series.value_counts().values)
        if 1 < number_of_categories < 100:
            data_series.value_counts().plot.density(ax=axarr[1])
    axarr[0].set_ylabel("Frequency")
    for ax in axarr:
        ax.set_yscale('log' if logy else 'linear')
        ax.set_xscale('log' if logx else 'linear')
    plt.suptitle('Histogram and KDE for {}'.format(data_series.name))

    plt.savefig("attributes/{}{}{}.pdf".format(data_series.name,
                                               "_logx" if logx else "",
                                               "_logy" if logy else ""), dpi=150)


def explore_data(file_name,budget_name="total_charge"):
    """ Print some stats and plot some stuff

    :param file_name: JSON file containing all data
    :type file_name: str
    :param budget_name: Use either "budget" or "total_charge"
    :type file_name: str
    """
    data_frame = prepare_data_raw(file_name)

    #for attr in data_frame.columns:
    #    plot_value_distributions(data_frame[attr])

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
