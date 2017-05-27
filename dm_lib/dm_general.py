from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt

def printDF(title, df):
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

def printCorr(df, attr=None):
    corr = df.corr()
    if attr==None:
        print "### Corrletaion Matrix ###"
        print corr
        plt.matshow(corr)
        plt.show()
    else:
        print "### Corrletaions for " + attr + " ###"
        print corr[attr].abs().sort_values(ascending=False)
    print "################################ \n\n"

def evaluateRegression(df, predictions, label_name):
    print "### Evaluation of " + label_name + " ###"
    exp_var_sc = explained_variance_score(df[label_name], predictions)
    print "## Explained variance score (best is 1.0): ", exp_var_sc
    abs_err = mean_absolute_error(df[label_name], predictions)
    print "## Mean absolute error: ", abs_err
    sq_err = mean_squared_error(df[label_name], predictions)
    print "## Mean squared error: ", abs_err

    return exp_var_sc, abs_err, sq_err

def evaluateClassification(df, predictions, label_name):
    print "### Evaluation of " + label_name + " ###"
    accuracy = accuracy_score(df[label_name], predictions)
    print "## Accuracy as a fraction: ", accuracy

    # TODO: add more measures http://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics

    return accuracy
