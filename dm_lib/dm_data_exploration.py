from dm_general import print_statistics, print_correlations
from dm_data_preparation import *
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


def perc_convert(ser):
  return ser/float(ser[-1])


def same_mean(series_1, series_2, significance):
    """ Make an independent T-Test for two series to check whether their means are the same

    :type series_1: pandas.Series
    :type series_2: pandas.Series
    :param significance: The significance is normally 5%

    """
    result = stats.ttest_rel(series_1, series_2)
    if result[1] <= significance:
        return True
    else:
        return False


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

    for attr in ["client_feedback", "timestamp", "freelancer_count", "workload", "total_hours"]:
        # data_frame[attr].plot(kind='hist', legend=True, title=attr)
        # data_frame.hist(column=attr, bins=30)
        chart, ax = plt.subplots()
        # sns.distplot(data_frame[attr], hist_kws={'cumulative': True}, kde_kws={'cumulative': True}, ax=ax)
        #stats.probplot(data_frame[attr], plot=ax)
        #data_frame.boxplot(column=attr, ax=ax)
        #data_frame.plot.scatter(x='total_charge', y=attr, logx=True)
        #sns.regplot(x=data_frame['freelancer_count'].apply(lambda count: np.log(count)), y=data_frame[attr].apply(lambda value: np.log(value)), ax=ax, order=0, lowess=True)
        sns.regplot(x=data_frame['freelancer_count'].apply(lambda count: np.log(count)),
                    y=data_frame[attr], ax=ax, order=0, lowess=True)
        #sns.residplot(x=data_frame['total_charge'], y=data_frame[attr], ax=ax, order=0, lowess=True)
        ax.set_title("LOWESS for " + attr)
        #plt.show()
    print_statistics(data_frame)
    print_correlations(data_frame, store=True)
