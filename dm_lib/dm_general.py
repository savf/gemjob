from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import os
from matplotlib.colors import ListedColormap

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


def print_correlations(df, attr=None, store=False, method='spearman',
                       xlabels=None, ylabels=None):
    """ Print attribute correlations for a given Pandas DataFrame

    Spearman and kendall are robust in regards to outliers whereas pearson
    can be affected by them and can give the wrong correlation as a result

    :param df: Pandas DataFrame to analyze
    :type df: pandas.DataFrame
    :param attr: If specified, only print correlations for the given attribute
    :type attr: str
    :param store: Whether to store the correlations and significance as CSV
    :type store: bool
    :param method: Which correlation method to use: pearson, spearman or kendall
    :type method: str
    :param xlabels: List with labels for the x-axis tick marks
    :type xlabels: list(str)
    :param ylabels: List with labels for the y-axis tick marks
    :type ylabels: list(str)
    """
    corr = df.corr(method)
    dropped_columns = list(set(df.columns) - set(corr.columns))
    df.drop(labels=dropped_columns, axis=1, inplace=True)
    significance = np.zeros([df.shape[1], df.shape[1]])

    for row in range(df.shape[1]):
        for column in range(df.shape[1]):
            row_label = df.columns[row]
            column_label = df.columns[column]
            if method == 'pearson':
                significance[row][column] = stats.pearsonr(df[row_label], df[column_label])[1]
            elif method == 'kendall':
                significance[row][column] = stats.kendalltau(df[row_label], df[column_label])[1]
            else:
                significance[row][column] = stats.spearmanr(df[row_label], df[column_label])[1]

    corr_significance = pd.DataFrame(significance)
    corr_significance.columns = df.columns.values
    corr_significance.set_index(df.columns.values, inplace=True)

    if attr is None:
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        plt.subplots_adjust(left=0.1, bottom=0.2)
        green_to_red = ["#30d43f", "#49ff61", "#ff8787", "#ff4545", "#ff0000"]
        custom_cmap = ListedColormap(sns.color_palette(green_to_red).as_hex())
        sns.heatmap(corr, vmax=1.0, square=True, cmap="OrRd", ax=ax1)
        sns.heatmap(significance, vmax=1.0, square=True, cmap=custom_cmap, ax=ax2)
        if xlabels is not None:
            ax1.set_xticklabels(xlabels)
            ax2.set_xticklabels(xlabels, rotation=90)
        if ylabels is not None:
            ax1.set_yticklabels(ylabels[::-1])
            ax2.set_yticklabels(ylabels[::-1], rotation=0)

    else:
        print "### Correlations for " + attr + " ###"
        print corr[attr].abs().sort_values(ascending=False)
    print "################################ \n\n"
    if store:
        with open('correlations.csv', 'w') as f:
            f.write(corr.to_csv())
        f.close()
        with open('correlation_significances.csv', 'w') as f:
            f.write(corr_significance.to_csv())
        f.close()


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
