import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas.plotting import scatter_matrix
import rethinkdb as rdb
from scipy import stats
from scipy.stats.mstats import normaltest
from sklearn.neighbors.kde import KernelDensity

from dm_data_preparation import *
from dm_general import print_correlations, print_statistics

RDB_HOST = '192.168.99.100'
RDB_PORT = 28015
RDB_DB = 'datasets'
RDB_OPTIMIZED_TABLE = 'jobs_optimized'


def perc_convert(ser):
    return ser/float(ser[-1])


def replace_missing_with_kde_samples(data_frame, attribute):
    """ Replace missing values based on samples from KDE function

    :param data_frame: Pandas dataframe holding the attribute
    :type data_frame: pandas.DataFrame
    :param attribute: The attribute for which missing values should be replaced
    :type attribute: str
    """
    minimum = data_frame[attribute].min()
    maximum = data_frame[attribute].max()
    values = np.array(data_frame[attribute].dropna())
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(
        values.reshape(-1, 1))
    missing_values = data_frame.loc[
        data_frame[attribute].isnull(), attribute]
    samples = [num for num in
               kde.sample(n_samples=len(data_frame[attribute].dropna()))
               if minimum <= num <= maximum]
    while len(samples) < 2*len(missing_values):
        samples.extend([num for num in
                        kde.sample(n_samples=len(data_frame[attribute].dropna()))
                        if minimum <= num <= maximum])
    samples = [samples[i] for i in
               sorted(random.sample(xrange(len(samples)), len(missing_values)))]

    for index, value in enumerate(samples):
        missing_values[index] = samples[index]

    data_frame.update(pd.DataFrame(missing_values))


def from_same_distribution(series1, series2, significance):
    """ Check whether two series stem from the same distribution

    :param series1: Pandas Series for first attribute
    :type series1: pandas.Series
    :param series2: Pandas Series for second attribute
    :type series2: pandas.Series
    :param significance: Test significance (normally 5%)
    :type significance: float
    :return: Whether same distribution, K-S Statistic and p-Value
    :rtype: bool, float, float
    """
    ks_statistic, p_value = stats.ks_2samp(series1, series2)

    # Small p-value means reject the null-hypothesis that the samples have the
    # same distribution
    if p_value <= significance:
        return False, ks_statistic, p_value
    else:
        return True, ks_statistic, p_value


def same_mean(series_1, series_2, significance):
    """ Check the variance and distribution and then make hypothesis test for the mean

    The variance and normal distribution test is needed to check whether a t-test could
    be used, since these two requirements are needed for the t-test. If these requirements
    are not met, the Mann-Whitney-Wilcoxon RankSum test is used

    :param series1: Pandas Series for first attribute
    :type series_1: pandas.Series
    :param series2: Pandas Series for second attribute
    :type series_2: pandas.Series
    :param significance: Test significance (normally 5%)
    :type significance: float
    :rtype: bool
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


def data_frame_from_db():
    connection = rdb.connect(RDB_HOST, RDB_PORT)
    jobs_cursor = rdb.db(RDB_DB).table(RDB_OPTIMIZED_TABLE).run(connection)
    jobs = list(jobs_cursor)
    data_frame = pd.DataFrame(jobs)
    data_frame.set_index('id', inplace=True)

    return data_frame

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


def plot_scatter_matrix(data_frame, attributes, logx=False, logy=False):
    """ Plot a scatter matrix with correctly rotated labels

    :param data_frame: DataFrame that holds the attributes
    :type data_frame: pandas.DataFrame
    :param attributes: List of attributes to plot
    :type attributes: list(str)
    :param logx: Set x axis to log for all scatter plots
    :type logx: bool
    :param logy: Set y axis to log for all scatter plots
    :type logy: bool
    """
    attributes_to_plot = [att for att in attributes if att in data_frame.columns]
    sm = scatter_matrix(data_frame[attributes_to_plot], alpha=0.2,
                        figsize=(6, 6), diagonal='kde')
    # Change label rotation
    [s.xaxis.label.set_rotation(90) for s in sm.reshape(-1)]
    [s.yaxis.label.set_rotation(0) for s in sm.reshape(-1)]
    [s.get_yaxis().set_label_coords(-1.5, 0.5) for s in sm.reshape(-1)]

    # Log scaling
    if logx:
        [s.set_xscale('log') for s in sm.reshape(-1)]
    if logy:
        [s.set_yscale('log') for s in sm.reshape(-1)]

    # Hide all ticks
    [s.set_xticks(()) for s in sm.reshape(-1)]
    [s.set_yticks(()) for s in sm.reshape(-1)]
    plt.subplots_adjust(bottom=0.25, left=0.55)
    plt.show()


def plot_qqplot(data_series, store=False):
    """ Create a QQ plot (aka probability plot) to check for normal dist.

    :param data_series: Panda Series containing the data to plot
    :type data_series: pandas.Series
    :param store: Whether to store the plot as PDF
    :type store: bool
    """
    fig, ax = plt.subplots()

    if np.issubdtype(get_datatype_safely(data_series.dtype), np.number):
        stats.probplot(data_series, plot=ax)
        ax.set_title("Probability plot for " + data_series.name)

        if store:
            plt.savefig("attributes/qq_{}.pdf".format(data_series.name),
                        dpi=150)


def plot_boxplot(data_series, store=False):
    """ Create a boxplot and save it as boxplot_attributename.pdf

    :param data_series: Pandas Series containing the data to plot
    :type data_series: pandas.Series
    :param store: Whether to store the plot as PDF
    :type store: bool
    """
    fig, ax = plt.subplots()

    if np.issubdtype(get_datatype_safely(data_series.dtype), np.number):
        sns.boxplot(data_series, ax=ax)

        plt.suptitle('Boxplot for {}'.format(data_series.name))

        if store:
            plt.savefig("attributes/boxplot_{}.pdf".format(data_series.name),
                        dpi=150)


def plot_value_distributions(data_series, x_label=None, logy=False, store=True):
    """ Plot a histogram, a KDE and a CDF for the given Series

    :param data_series: The data series to be plotted
    :type data_series: pandas.Series
    :param x_label: Label to show on histogram, KDE and CDF x-Axis
    :type x_label: str
    :param logy: Plot the y axis for the histogram in log scale
    :type logy: bool
    :param store: Whether to store the generated plot
    :type store: bool
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

    if store:
        plt.savefig("attributes/{}{}.pdf".format(data_series.name,
                                                 "_logy" if logy else ""),
                    dpi=150)
    else:
        plt.show()


def explore_data(file_name,budget_name="total_charge"):
    """ Print some stats and plot some stuff

    :param file_name: JSON file containing all data
    :type file_name: str
    :param budget_name: Use either "budget" or "total_charge"
    :type file_name: str
    """
    # data_frame = prepare_data_raw(file_name)

    feedbacks_for_client = ['feedback_for_client_availability',
                            'feedback_for_client_communication',
                            'feedback_for_client_cooperation',
                            'feedback_for_client_deadlines',
                            'feedback_for_client_quality',
                            'feedback_for_client_skills']

    feedbacks_for_freelancer = ['feedback_for_freelancer_availability',
                                'feedback_for_freelancer_communication',
                                'feedback_for_freelancer_cooperation',
                                'feedback_for_freelancer_deadlines',
                                'feedback_for_freelancer_quality',
                                'feedback_for_freelancer_skills']

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

    #
    # for feedback in feedbacks_for_freelancer:
    #     for feedback2 in feedbacks_for_freelancer:
    #         result, ks_statistic, p_value = from_same_distribution(
    #             data_frame[feedback], data_frame[feedback2], 0.05)
    #         print "{},{},{},{},{}".format(feedback, feedback2,
    #                                       result, ks_statistic, p_value)
    #
    # data_frame["feedback_for_client"] = data_frame[
    #     ['feedback_for_client_availability',
    #      'feedback_for_client_cooperation',
    #      'feedback_for_client_deadlines']].mean(axis=1)
    # data_frame["feedback_for_freelancer"] = data_frame[
    #     ['feedback_for_freelancer_availability',
    #      'feedback_for_freelancer_cooperation',
    #      'feedback_for_freelancer_quality',
    #      'feedback_for_freelancer_skills']].mean(axis=1)
    #
    # feedbacks_for_client = ['feedback_for_client_availability',
    #                         'feedback_for_client_cooperation',
    #                         'feedback_for_client_deadlines']
    #
    # feedbacks_for_freelancer = ['feedback_for_freelancer_availability',
    #                             'feedback_for_freelancer_cooperation',
    #                             'feedback_for_freelancer_quality',
    #                             'feedback_for_freelancer_skills']
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

    # feedbacks_for_client_frame = data_frame[[col for col
    #                                          in data_frame.columns
    #                                          if col.startswith('feedback_for_client_')]]
    #
    # print_correlations(feedbacks_for_client_frame.dropna(),
    #                    store=True, method='pearson',
    #                    xlabels=[col.split('_')[3] for col in
    #                             feedbacks_for_client_frame.columns],
    #                    ylabels=[col.split('_')[3] for col in
    #                             feedbacks_for_client_frame.columns])

    # data_frame = get_overall_job_reviews(data_frame, drop_detailed=False)

    # plot_value_distributions(data_frame['feedback_for_client'], x_label="feedback score", logy=True)
    # plot_value_distributions(data_frame['feedback_for_freelancer'], x_label="feedback score", logy=True)
    # plot_boxplot(data_frame['feedback_for_client'], store=True)
    # plot_boxplot(data_frame['feedback_for_freelancer'], store=True)

    # print_statistics(data_frame[['feedback_for_client', 'feedback_for_freelancer']])

    # for aggregate_feedback in ['feedback_for_client', 'feedback_for_freelancer']:
    #     for feedback in feedbacks_for_client if aggregate_feedback ==\
    #             'feedback_for_client' else feedbacks_for_freelancer:
    #         print "{},{},{},{}".format(aggregate_feedback, feedback,
    #                                    data_frame[aggregate_feedback].var(),
    #                                    data_frame[feedback].var())

    # attribute = 'feedback_for_freelancer'
    # forwardfill = data_frame[attribute].fillna(method='pad')
    # backfill = data_frame[attribute].fillna(method='backfill')
    # kdesamples = data_frame.copy()  # type: pd.DataFrame
    # replace_missing_with_kde_samples(kdesamples, attribute)
    # meanfill = data_frame[attribute].fillna(data_frame[attribute].mean())
    #
    # print "{} {}:\r\n{}".format(attribute, "forwardfill", forwardfill.describe())
    # print "{} {}:\r\n{}".format(attribute, "backfill", backfill.describe())
    # print "{} {}:\r\n{}".format(attribute, "kdesamples", kdesamples[attribute].describe())
    # print "{} {}:\r\n{}".format(attribute, "meanfill", meanfill.describe())
    #
    # print from_same_distribution(data_frame[attribute].dropna(), forwardfill, 0.05)
    # print from_same_distribution(data_frame[attribute].dropna(), backfill, 0.05)
    # print from_same_distribution(data_frame[attribute].dropna(), kdesamples[attribute], 0.05)
    # print from_same_distribution(data_frame[attribute].dropna(), meanfill, 0.05)

    # pd.set_option('display.float_format', lambda x: '%.6f' % x)
    #
    # data_frame.loc[data_frame['client_payment_verification_status'].isnull(),
    #                'client_payment_verification_status'] = "UNKNOWN"
    # verified = data_frame.loc[data_frame['client_payment_verification_status'] == "VERIFIED"]
    # allelse = data_frame.loc[data_frame['client_payment_verification_status'] != "VERIFIED"]
    #
    # for attribute in ["budget", "total_charge", "client_feedback"]:
    #     print "{} SameMean: {}".format(attribute, same_mean(verified[attribute].dropna(),
    #                                                         allelse[attribute].dropna(),
    #                                                         0.05))
    #     print "{} SameDistribution: {}".format(attribute,
    #                                            from_same_distribution(verified[attribute].dropna(),
    #                                                                   allelse[attribute].dropna(),
    #                                                                   0.05))
    #     print "verified:"
    #     print verified[attribute].describe()
    #     print "allelse:"
    #     print allelse[attribute].describe()
    #     print "\r\n"
    #
    # cleandf = data_frame[["duration_weeks_total", "duration_weeks_median"]].dropna()
    # for method in ["spearman", "kendall", "pearson"]:
    #     print cleandf.corr(method=method)
    # print stats.spearmanr(cleandf['duration_weeks_total'],
    #                        cleandf['duration_weeks_median'])
    # print stats.kendalltau(cleandf['duration_weeks_total'],
    #                        cleandf['duration_weeks_median'])
    # print stats.pearsonr(cleandf['duration_weeks_total'],
    #                        cleandf['duration_weeks_median'])
    #
    # q1 = data_frame.quantile(0.25)
    # q3 = data_frame.quantile(0.75)
    # iqr = q3 - q1
    #
    # print ((data_frame < (q1 - 1.5 * iqr)) | (data_frame > (q3 + 1.5 * iqr))).sum()

    # attribute = 'experience_level'
    # forwardfill = data_frame[attribute].fillna(method='pad')
    # backfill = data_frame[attribute].fillna(method='backfill')
    # meanfill = data_frame[attribute].fillna(data_frame[attribute].value_counts().index[0])
    #
    # print "{} {}:\r\n{}".format(attribute, "forwardfill", forwardfill.describe())
    # print "{} {}:\r\n{}".format(attribute, "backfill", backfill.describe())
    # print "{} {}:\r\n{}".format(attribute, "meanfill", meanfill.describe())
    #
    # print from_same_distribution(data_frame[attribute].dropna(), forwardfill, 0.05)
    # print from_same_distribution(data_frame[attribute].dropna(), backfill, 0.05)
    # print from_same_distribution(data_frame[attribute].dropna(), meanfill, 0.05)

    # for attribute in data_frame.columns:
    #     plot_qqplot(data_frame[attribute], store=True)

    # data_frame = prepare_data(file_name)
    # data_frame.date_created = data_frame.date_created.apply(lambda time: time.to_pydatetime().replace(
    #     tzinfo=rdb.make_timezone("+02:00")))
    #
    # data_frame['id'] = data_frame.index
    #
    # connection = rdb.connect(RDB_HOST, RDB_PORT)
    # response = rdb.db(RDB_DB).table(RDB_OPTIMIZED_TABLE).insert(
    #     data_frame.to_dict('records'), conflict="replace").run(connection)
    # connection.close()

    data_frame = load_data_frame_from_db()
    data_frame.drop(labels=['client_country'], axis=1, inplace=True)
    data_frame = convert_to_numeric(data_frame, None)

    data_frame_hourly = data_frame.loc[data_frame['job_type_Hourly'] == 1]
    data_frame_hourly.drop(labels=['job_type_Fixed',
                                   'job_type_Hourly',
                                   'budget'], axis=1, inplace=True)
    data_frame_fixed = data_frame.loc[data_frame['job_type_Fixed'] == 1]
    data_frame_fixed.drop(labels=['job_type_Fixed',
                                  'job_type_Hourly',
                                  'workload'], axis=1, inplace=True)
    # print_correlations(data_frame_hourly, store=True,
    #                    xlabels=data_frame_hourly.columns.values,
    #                    ylabels=data_frame_hourly.columns.values)
    # print_correlations(data_frame_fixed, store=True,
    #                    xlabels=data_frame_fixed.columns.values,
    #                    ylabels=data_frame_fixed.columns.values)

    # plot_scatter_matrix(data_frame_fixed, ['budget',
    #                                         'client_feedback',
    #                                         'client_jobs_posted',
    #                                         'client_past_hires',
    #                                         'client_reviews_count',
    #                                         'duration_weeks_median',
    #                                         'feedback_for_client',
    #                                         'feedback_for_freelancer',
    #                                         'freelancer_count',
    #                                         'total_charge', 'total_hours'])
