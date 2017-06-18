from dm_general import print_statistics, print_correlations
from dm_data_preparation import *

def percConvert(ser):
  return ser/float(ser[-1])

def exploreData(file_name,budget_name="total_charge"):
    """ Print some stats and plot some stuff

    :param file_name: JSON file containing all data
    :type file_name: str
    :param budget_name: Use either "budget" or "total_charge"
    :type file_name: str
    """
    data_frame = prepare_data(file_name, budget_name=budget_name)

    print "## crosstab job_type vs experience_level"
    print pd.crosstab(data_frame["job_type"], data_frame["experience_level"], margins=True).apply(percConvert, axis=1)

    drop_unnecessary = ["client_country"]
    data_frame.drop(labels=drop_unnecessary, axis=1, inplace=True)

    data_frame = convert_to_numeric(data_frame, "")

    for attr in ["client_feedback", "timestamp", "freelancer_count", "workload", "total_hours"]:
        # data_frame[attr].plot(kind='hist', legend=True, title=attr)
        data_frame.hist(column=attr, bins=30)
    print_statistics(data_frame)
    print_correlations(data_frame)