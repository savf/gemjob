import os
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import explained_variance_score, mean_squared_error, \
    mean_absolute_error, accuracy_score


def block_printing():
    sys.stdout = open(os.devnull, 'w')


def enable_printing():
    sys.stdout = sys.__stdout__


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
    pd.set_option('display.max_rows', num_of_rows)
    if len(df) != len(predictions):
        print "\n### Error: Length of values does not match\n"
        return
    print "\n\n### Compare predictions to actual: ###\n"
    df['predictions'] = predictions
    print df[["predictions", label_name]][0:num_of_rows]
    print "###########\n\n"
    pd.reset_option('display.max_rows')


def mape_vectorized(a, b):
    mask = a != 0
    return (np.fabs(a - b)*100/a)[mask].mean()


def evaluate_regression(df, predictions, label_name, printing=True):
    """ Print explained variance, mean absolute error and mean squared error for given regression results

    :param df: Pandas DataFrame containing the data
    :type df: pandas.DataFrame
    :param predictions: Array holding the regression predictions
    :type predictions: array
    :param label_name: Target label
    :type label_name: str
    :param printing: Whether to print to console
    :type printing: bool
    """
    if not printing:
        block_printing()

    print "### Evaluation of " + label_name + " ###"
    exp_var_sc = explained_variance_score(df, predictions)
    print "## Explained variance score (best is 1.0): ", exp_var_sc
    abs_err = mean_absolute_error(df, predictions)
    print "## Mean absolute error: ", abs_err
    sq_err = mean_squared_error(df, predictions)
    print "## Mean squared error: ", sq_err
    mape = mape_vectorized(df, predictions)
    print "## Mean absolute percentage error: {:.1f}%".format(mape)
    print ""

    enable_printing()
    return exp_var_sc, abs_err, sq_err, mape


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


def evaluate_classification(df, predictions, label_name, printing=True):
    """ Print accuracy of classification

    :param df: Pandas DataFrame containing the data
    :type df: pandas.DataFrame
    :param predictions: Array holding the classification predictions
    :type predictions: array
    :param label_name: Target label
    :type label_name: str
    :param printing: Whether to print to console
    :type printing: bool
    """
    if not printing:
        block_printing()
    if isinstance(df, pd.DataFrame):
        df = df[label_name]
    print "### Evaluation of " + label_name + " ###"
    accuracy = accuracy_score(df, predictions)
    print "## Accuracy as a fraction: ", accuracy
    print ""

    # TODO: add more measures http://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics

    enable_printing()
    return accuracy


def print_model_evaluation(model, df_test, label_name,
                           is_classification, csv=False):
    """ Print accuracy of model

    :param df_test: Pandas DataFrame containing the test data
    :type df_test: pandas.DataFrame
    :param label_name: Target label
    :type label_name: str
    :param is_classification: classification or regression?
    :type is_classification: bool
    :param csv: Print comma separated to use in spreadsheets
    :type csv: bool
    """
    # separate target
    df_target_test = df_test[label_name]
    df_test.drop(labels=[label_name], axis=1, inplace=True)
    # predict
    predictions = model.predict(df_test)
    results = []

    if is_classification:
        accuracy = evaluate_classification(
            df_target_test, predictions, label_name, printing=not csv)
        results.append(accuracy)
    else:
        exp_var_sc, abs_err, sq_err, mape = evaluate_regression(
            df_target_test, predictions, label_name, printing=not csv)
        results.extend([exp_var_sc, abs_err, sq_err, mape])

    if not csv:
        print_predictions_comparison(pd.DataFrame({label_name: df_target_test}),
                                     predictions, label_name, num_of_rows=20)
    return results


def generate_model_stats(data_frame, model):
    feature_importances = pd.DataFrame(columns=data_frame.columns)
    for estimator in model.estimators_:
        importances = pd.DataFrame([estimator.feature_importances_],
                                   columns=data_frame.columns)
        feature_importances = feature_importances.append(importances,
                                                         ignore_index=True)

    feature_importances_mean = feature_importances.mean().sort_values(ascending=False)

    text = {'text': [], 'title': [], 'snippet': [], 'skills': []}
    engineered_text = ['snippet_length', 'skills_number', 'title_length']
    categorical = {'client_country': [], 'subcategory2': [], 'experience_level': [],
                   'job_type': []}
    numerical = {}

    for column in feature_importances_mean.keys():
        if feature_importances_mean.loc[column] > 0:
            if column.startswith('$token') or column in engineered_text:
                text['text'].append(column)
                for text_element in ['title', 'snippet', 'skills']:
                    if column.startswith('$token_' + text_element):
                        text[text_element].append(column)
                if column in engineered_text:
                    text[column] = [column]
            else:
                if column.startswith(tuple(categorical.keys())):
                    for key, value in categorical.iteritems():
                        if column.startswith(key):
                            categorical[key].append(column)
                else:
                    numerical[column] = [column]

    importances = text.copy()
    importances.update(categorical)
    importances.update(numerical)
    importances_iteration = importances.copy()
    for key, value in importances_iteration.iteritems():
        column_indices = [data_frame.columns.get_loc(column) for column in
                          importances[key]]
        std = np.std([sub_model.feature_importances_[column_indices].sum()
                      for sub_model in model.estimators_], axis=0)
        importance = feature_importances_mean.loc[importances[key]].sum()
        importances[key] = {'importance': round(importance*100, 2),
                            'error': [round(max(0.0, min(importance - std, 1.0))*100, 2),
                                      round(max(0.0, min(importance + std, 1.0))*100, 2)],
                            'std': round(std*100, 2)}
        if importances[key]['importance'] == 0.0:
            del importances[key]

    return importances


def evaluate_predictions_regression(function, function_args, iterations=10):
    """ Get average accuracy of a regression model after many iterations

    :param function: Function returning the result of the evaluate_regression function in the end
    :type function: def
    :param function_args: Arguments for the function
    :type function_args: various
    :return: Average errors
    :rtype: various
    """

    if iterations == 0: return

    tot_exp_var_sc = 0
    tot_abs_err = 0
    tot_sq_err = 0
    tot_mape = 0

    for i in range(0, iterations):
        exp_var_sc, abs_err, sq_err, mape = function(*function_args)
        tot_exp_var_sc = tot_exp_var_sc + exp_var_sc
        tot_abs_err = tot_abs_err + abs_err
        tot_sq_err = tot_sq_err + sq_err
        tot_mape = tot_mape + mape

    tot_exp_var_sc = tot_exp_var_sc / iterations
    tot_abs_err = tot_abs_err / iterations
    tot_sq_err = tot_sq_err / iterations
    tot_mape = tot_mape / iterations

    print "\n\n\n\n####### Final average evaluation after", iterations, "iterations #######"
    print "## Explained variance score (best is 1.0): ", tot_exp_var_sc
    print "## Mean absolute error: ", tot_abs_err
    print "## Mean squared error: ", tot_sq_err
    print "## Mean absolute percentage error: {:.1f}%".format(tot_mape)

    return tot_exp_var_sc, tot_abs_err, tot_sq_err, tot_mape


def evaluate_predictions_classification(function, function_args, iterations=10):
    """ Get average accuracy of a classification model after many iterations

    :param function: Function returning the result of the evaluate_classification function in the end
    :type function: def
    :param function_args: Arguments for the function
    :type function_args: various
    :return: Average errors
    :rtype: various
    """

    if iterations == 0: return

    tot_accuracy = 0

    for i in range(0, iterations):
        accuracy = function(*function_args)
        tot_accuracy = tot_accuracy + accuracy

    tot_accuracy = tot_accuracy / iterations

    print "\n\n\n\n####### Final average evaluation after", iterations, "iterations #######"
    print "## Accuracy: ", tot_accuracy

    return tot_accuracy
