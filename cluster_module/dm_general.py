import os

import pandas as pd
import scipy.stats as stats
from sklearn.metrics import explained_variance_score, mean_squared_error, \
    mean_absolute_error, accuracy_score


def print_data_frame(title, df):
    """ Print stats about a given Pandas DataFrame with a given title

    :param title: Title to be displayed in the printout
    :type title: str
    :param df: Pandas DataFrame to analyze
    :type df: pandas.DataFrame
    """
    print "##############################\n    "+title+"    \n##############################\n"
    print "## Shape: ##"
    print df.shape
    print "\n## Missing Values per Column: ##"
    print df.isnull().sum()
    print "\n## Data Types: ##"
    print df.dtypes
    # print "\n## Show data: ##"
    # print df[0:5]
    print "############################## \n\n"


def print_correlations(df, attr=None, method='spearman'):
    """ Print attribute correlations for a given Pandas DataFrame

    Spearman and kendall are robust in regards to outliers whereas pearson
    can be affected by them and can give the wrong correlation as a result

    :param df: Pandas DataFrame to analyze
    :type df: pandas.DataFrame
    :param attr: If specified, only print correlations for the given attribute
    :type attr: str
    :param method: Correlation method to use (spearman, kendall or pearson)
    :type method: str
    """
    corr = df.corr(method)

    if attr is not None:
        print "### Correlations for " + attr + " ###"
        print corr[attr].abs().sort_values(ascending=False)
        print "### p-Values for correlations for " + attr + " ###"
        if method == "spearman":
            for column in range(df.shape[1]):
                print "{} and {}: {}".format(attr,
                                             df.columns[column],
                                             stats.spearmanr(df[attr], df[df.columns[column]]))
        elif method == "kendall":
            for column in range(df.shape[1]):
                print "{} and {}: {}".format(attr,
                                             df.columns[column],
                                             stats.kendalltau(df[attr], df[df.columns[column]]))
        else:
            for column in range(df.shape[1]):
                print "{} and {}: {}".format(attr,
                                             df.columns[column],
                                             stats.pearsonr(df[attr], df[df.columns[column]]))
    print "################################ \n\n"


def print_statistics(df):
    print "\n### Statistics ###\n"
    if not os.path.exists("attribute_statistics"):
        os.makedirs("attribute_statistics")
    for attribute in df.columns:
        with open("attribute_statistics/{}.txt".format(attribute), mode='w') as f:
            f.write("name: " + attribute + "\n")
            f.write("missing: {}\n".format(df[attribute].isnull().sum()))
            values = dict(df[attribute].describe())
            for key,value in values.iteritems():
                f.write("{}: {}\n".format(key, value))
            f.write("type: {}".format(str(df[attribute].dtype)))
        f.close()

    print "################################ \n\n"


def print_predictions_comparison(df, predictions, label_name, num_of_rows=10):
    """ Print predictions next to actual values

    :param df: Pandas DataFrame containing the data
    :type df: pandas.DataFrame
    :param predictions: Array holding the predictions
    :type predictions: array
    :param label_name: Target label
    :type label_name: str
    :param num_of_rows: Number of rows to diplay
    :type num_of_rows: int
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df, columns=[label_name])
    pd.set_option('display.max_rows', num_of_rows)
    if len(df) != len(predictions):
        print "\n### Error: Length of values does not match\n"
        return
    print "\n\n### Compare predictions to actual: ###\n"
    df['predictions'] = predictions
    print df[["predictions", label_name]][0:num_of_rows]
    print "###########\n\n"
    pd.reset_option('display.max_rows')


def evaluate_regression(df, predictions, label_name):
    """ Print explained variance, mean absolute error and mean squared error for given regression results

    :param df: Pandas DataFrame containing the data
    :type df: pandas.DataFrame
    :param predictions: Array holding the regression predictions
    :type predictions: array
    :param label_name: Target label
    :type label_name: str
    """
    if isinstance(df, pd.DataFrame):
        df = df[label_name]
    print "### Evaluation of " + label_name + " ###"
    exp_var_sc = explained_variance_score(df, predictions)
    print "## Explained variance score (best is 1.0): ", exp_var_sc
    abs_err = mean_absolute_error(df, predictions)
    print "## Mean absolute error: ", abs_err
    sq_err = mean_squared_error(df, predictions)
    print "## Mean squared error: ", sq_err

    return exp_var_sc, abs_err, sq_err


def evaluate_regression_csv(df, predictions, label_name, predicted_with_attribute, model_name, parameters, runtime):
    """ Evaluate the regression and output the evaluation as a CSV line

    :param df: Pandas DataFrame containing the data
    :type df: pandas.DataFrame
    :param predictions: Array holding the regression predictions
    :type predictions: array
    :param label_name: Target label
    :type label_name: str
    :param predicted_with_attribute: Attribute that was used to predict the target label
    :type predicted_with_attribute: str
    :param model_name: Name of the regression model used for the prediction
    :type model_name: str
    :param parameters: Parameters used in the regression model
    :type parameters: dict
    :param runtime: Runtime of the model in seconds
    :type runtime: float
    """
    if isinstance(df, pd.DataFrame):
        df = df[label_name]
    exp_var_sc = explained_variance_score(df, predictions)
    abs_err = mean_absolute_error(df, predictions)
    sq_err = mean_squared_error(df, predictions)
    parameter_string = ",".join([str(value) for key, value in parameters.iteritems()])

    print label_name+","+predicted_with_attribute+","+model_name+","+parameter_string+","\
        + str(exp_var_sc)+","+str(abs_err)+","+str(sq_err)+","+str(runtime)


def evaluate_classification(df, predictions, label_name):
    """ Print accuracy of classification

    :param df: Pandas DataFrame containing the data
    :type df: pandas.DataFrame
    :param predictions: Array holding the classification predictions
    :type predictions: array
    :param label_name: Target label
    :type label_name: str"""
    if isinstance(df, pd.DataFrame):
        df = df[label_name]
    print "### Evaluation of " + label_name + " ###"
    accuracy = accuracy_score(df, predictions)
    print "## Accuracy as a fraction: ", accuracy

    # TODO: add more measures http://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics

    return accuracy


def print_model_evaluation(model, df_test, label_name, is_classification):
    """ Print accuracy of model

    :param df_test: Pandas DataFrame containing the test data
    :type df_test: pandas.DataFrame
    :param label_name: Target label
    :type label_name: str
    :param is_classification: classification or regression?
    :type is_classification: bool"""
    print "\n########## Evaluate model\n"
    # separate target
    df_target_test = df_test[label_name]
    df_test.drop(labels=[label_name], axis=1, inplace=True)
    # predict
    predictions = model.predict(df_test)

    if is_classification:
        evaluate_classification(df_target_test, predictions, label_name)
    else:
        evaluate_regression(df_target_test, predictions, label_name)

    print_predictions_comparison(df_target_test, predictions, label_name, 20)
