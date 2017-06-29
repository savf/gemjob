import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.core.dtypes.dtypes import CategoricalDtype
from scipy import stats
from scipy.stats.mstats import normaltest

from dm_data_preparation import *
from dm_general import print_correlations


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
        return False, result, p_value
    else:
        return True, result, p_value


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
    data_frame['experience_level'] = data_frame['experience_level'].astype(object)

    return data_frame


def get_datatype_safely(dtype):
    if isinstance(dtype, CategoricalDtype):
        return np.object_
    return dtype


def plot_boxplot(data_series, store=False):
    """ Create a boxplot and save it as boxplot_attributename.pdf

    :param data_series: Pandas Series containing the data to plot
    :type data_series: pandas.Series
    :param store: Whether to store the plot as PDF
    :type store: bool
    """
    fix, ax = plt.subplots()

    if np.issubdtype(get_datatype_safely(data_series.dtype), np.number):
        sns.boxplot(data_series, ax=ax)

        plt.suptitle('Boxplot for {}'.format(data_series.name))

        plt.savefig("attributes/boxplot_{}.pdf".format(data_series.name),
                    dpi=150)


def plot_value_distributions(data_series, x_label=None, logy=False):
    """ Plot a histogram, a KDE and a CDF for the given Series

    :param data_series: The data series to be plotted
    :type data_series: pandas.Series
    :param x_label: Label to show on histogram, KDE and CDF x-Axis
    :type x_label: str
    :param logy: Plot the y axis for the histogram in log scale
    :type logy: bool
    """
    # Check if data is numeric or not
    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=1, ncols=4, width_ratios=[1, 3, 3, 3])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlabel(x_label if x_label is not None else data_series.name)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_xlabel(x_label if x_label is not None else data_series.name)
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.set_xlabel(x_label if x_label is not None else data_series.name)
    plt.subplots_adjust(wspace=0.70, bottom=0.4, left=0.1, right=0.97, top=0.9)

    if np.issubdtype(get_datatype_safely(data_series.dtype), np.number):
        pd.Series({'missing': len(data_series.loc[data_series.isnull()])})\
            .plot(kind='bar', ax=ax1)
        ax1.set_ylabel("Frequency")
        toplot = data_series.dropna()
        toplot.hist(bins=30, ax=ax2)
        toplot.plot.density(ax=ax3)
        #sns.kdeplot(toplot, cumulative=True, legend=False, ax=ax4)
        toplot.hist(cumulative=True, normed=1, bins=100,
                    histtype='step', linewidth=2, ax=ax4)
        ax4.set_ylabel("Cumulative Probability")

        plt.suptitle('Histogram, KDE and CDF for {}'.format(data_series.name))
    else:
        data_series = data_series.astype(object)
        ax1.axis('off')
        ax3.axis('off')
        ax4.axis('off')
        toplot = data_series.value_counts()\
            .append(pd.Series({'missing': len(data_series.loc[data_series.isnull()])}))
        toplot.plot(kind='bar', ax=ax2)

        plt.suptitle('Histogram for {}'.format(data_series.name))

    ax2.set_ylabel("Frequency{}".format(" (log)" if logy else ""))
    ax2.set_yscale('log' if logy else 'linear')

    plt.savefig("attributes/{}{}.pdf".format(data_series.name,
                                             "_logy" if logy else ""),
                dpi=150)


def explore_data(file_name,budget_name="total_charge"):
    """ Print some stats and plot some stuff

    :param file_name: JSON file containing all data
    :type file_name: str
    :param budget_name: Use either "budget" or "total_charge"
    :type file_name: str
    """
    data_frame = prepare_data_raw(file_name)

    # dont_consider = ['title', 'snippet', 'skills', 'url', 'date_created']
    #
    # attributes = set(data_frame.columns).difference(dont_consider)
    #
    # fig, ax = plt.subplots()
    #
    # for attr in attributes:
    #     #plot_boxplot(data_frame[attr], store=True)
    #     plot_value_distributions(data_frame[attr], logy=True)

    # detailed_feedbacks_names = get_detailed_feedbacks_names()
    #
    # for feedback in detailed_feedbacks_names:
    #     plot_value_distributions(data_frame[feedback],
    #                              x_label=feedback.split('_')[3],
    #                              logy=False)
    #     plot_value_distributions(data_frame[feedback],
    #                              x_label=feedback.split('_')[3],
    #                              logy=True)

    # feedbacks_for_client = ['feedback_for_client_availability',
    #                         'feedback_for_client_communication',
    #                         'feedback_for_client_cooperation',
    #                         'feedback_for_client_deadlines',
    #                         'feedback_for_client_quality']
    #
    # feedbacks_for_freelancer = ['feedback_for_freelancer_availability',
    #                             'feedback_for_freelancer_communication',
    #                             'feedback_for_freelancer_cooperation',
    #                             'feedback_for_freelancer_deadlines',
    #                             'feedback_for_freelancer_quality',
    #                             'feedback_for_freelancer_skills']
    #
    # for feedback in feedbacks_for_client:
    #     for feedback2 in feedbacks_for_client:
    #         result, f_stat, p_value = same_mean(data_frame[feedback].dropna(),
    #                                             data_frame[feedback2].dropna(),
    #                                             0.05)
    #         print "{},{},{},{},{}".format(feedback, feedback2,
    #                                       result, f_stat, p_value)

    # data_frame = get_overall_job_reviews(data_frame, drop_detailed=False)
    #
    # for feedback in feedbacks_for_freelancer:
    #     result, f_stat, p_value = same_mean(data_frame['feedback_for_freelancer'].dropna(),
    #                                         data_frame[feedback].dropna(), 0.05)
    #     print "{},{},{},{},{}".format('feedback_for_freelancer', feedback,
    #                                   result, f_stat, p_value)
    #
    # for feedback in feedbacks_for_client:
    #     result, f_stat, p_value = same_mean(data_frame['feedback_for_client'].dropna(),
    #                                         data_frame[feedback].dropna(), 0.05)
    #     print "{},{},{},{},{}".format('feedback_for_client', feedback,
    #                                   result, f_stat, p_value)

    print_correlations(data_frame, store=True, method='pearson')

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
    print_correlations(data_frame, store=True)
