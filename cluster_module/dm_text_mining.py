from dm_general import evaluate_classification, evaluate_regression, print_predictions_comparison
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn import svm
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_random_state
import time
import pandas as pd


def clean_text(df, text_column_name):
    """ Perform stemming, stop word removal and lowercasing

    :param df: Pandas DataFrame containing the data
    :type df: pandas.DataFrame
    :param text_column_name: Column in data frame that contains the text values
    :type text_column_name: str
    :return: Cleaned data frame
    :rtype: pandas.DataFrame
    """
    is_str = isinstance(df[text_column_name].values[0], basestring)
    if is_str:
        stop_words = set(stopwords.words("english"))
        ps = PorterStemmer()
        df[text_column_name] = df.apply(lambda row: ' '.join(
            [ps.stem(w).lower() for w in word_tokenize(row[text_column_name]) if not w in stop_words]), axis=1)
    else:
        df[text_column_name] = df.apply(lambda row: ' '.join([w.lower() for w in row[text_column_name]]), axis=1)

    return df


def prepare_text_train(df, text_column_name, max_features):
    """ Create bag of words of the training dataset

    :param df: Pandas DataFrame containing the data
    :type df: pandas.DataFrame
    :param text_column_name: Attribute that contains the text values
    :type text_column_name: str
    :param max_features: Maximum number of features that will be produced by the vectorizer
    :type max_features: int
    :return: Both the vectorizer and the extracted features
    :rtype: sklearn.feature_extraction.text.CountVectorizer, array
    """
    df = clean_text(df, text_column_name)

    # bag of words
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,
                                 max_features=max_features)
    train_data_features = vectorizer.fit_transform(df[text_column_name])
    train_data_features = train_data_features.toarray()

    return vectorizer, train_data_features


def prepare_text_test(df, text_column_name, vectorizer):
    """ Create bag of words of the test dataset

        :param df: Pandas DataFrame containing the data
        :type df: pandas.DataFrame
        :param text_column_name: Attribute that contains the text values
        :type text_column_name: str
        :param max_features: Maximum number of features that will be produced by the vectorizer
        :type max_features: int
        :return: The features extracted by the CountVectorizer
        :rtype: array
        """
    df = clean_text(df, text_column_name)

    # bag of words
    test_data_features = vectorizer.transform(df[text_column_name])
    test_data_features = test_data_features.toarray()

    return test_data_features


def text_regression_model(df, label_name, train_data_features):
    """ Learn a SVM based regression model for the given target label

    :param df: Pandas DataFrame containing the data
    :type df: pandas.DataFrame
    :param label_name: Target label to be learned
    :type label_name: str
    :param train_data_features: Bag of words
    :type train_data_features: array
    :return: Learned regression model
    :rtype: sklearn.svm.SVR
    """
    clf = svm.SVR(kernel='linear') #RandomForestRegressor() #BaggingRegressor() #linear_model.LinearRegression()
    clf.fit(train_data_features, df[label_name])
    return clf


def text_classification_model(df, label_name, train_data_features, n_estimators=100):
    """ Lean a random forest classifier model for the given target label

    :param df: Pandas DataFrame containing the data
    :type df: pandas.DataFrame
    :param label_name: Target label to be learned
    :type label_name: str
    :param train_data_features: Bag of words
    :type train_data_features: array
    :param n_estimators: Number of trees in the forest
    :type n_estimators: int
    :return: Learned random forest model
    :rtype: sklearn.ensemble.RandomForestClassifier
    """
    # random forest classifier
    clf = RandomForestClassifier(n_estimators=n_estimators)
    # # naive bayes -> way worse
    # clf = MultinomialNB()
    clf.fit(train_data_features, df[label_name])
    return clf


def do_text_mining(text_train, text_test, label_name, regression, max_features=5000):
    """ Vectorize the given data frames, learn a regression or classification and print the results

    :param text_train: Pandas DataFrame containing the training data
    :type text_train: pandas.DataFrame
    :param text_test: Pandas DataFrame containing the test data
    :type text_test: pandas.DataFrame
    :param label_name: Target label to be used for classification or regression
    :type label_name: str
    :param regression: Boolean signifying whether regression or classification should be performed
    :type regression: bool
    :param max_features: Maximum size for bag of words
    :type max_features: int
    :return: Array with information about the prediction
    :rtype: array
    """
    print "##############################\n    Text Mining for " + label_name + \
          "    \n##############################\n"

    text_columns=["skills", "title", "snippet"]
    for text_column_name in text_columns:
        print "### Predict ", label_name, " with: ", text_column_name, " ###"

        vectorizer, train_data_features = prepare_text_train(text_train, text_column_name, max_features)
        test_data_features = prepare_text_test(text_test, text_column_name, vectorizer)
        if regression:
            model = text_regression_model(text_train, label_name, train_data_features)
        else:
            model = text_classification_model(text_train, label_name, train_data_features)

        predictions = model.predict(test_data_features)

        # evaluate
        if regression:
            evaluate_regression(text_test, predictions, label_name)
        else:
            evaluate_classification(text_test, predictions, label_name)

        print_predictions_comparison(text_test, predictions, label_name)

    print "################################ \n\n"
    return predictions


def addTextTokensToDF(df_train, df_test, text_train, text_test, max_features=500):
    """ Add tokenized text to data frame

        :param df_train: Pandas DataFrame containing the structured training data
        :type df_train: pandas.DataFrame
        :param df_test: Pandas DataFrame containing the structured test data
        :type df_test: pandas.DataFrame
        :param text_train: Pandas DataFrame containing the text training data
        :type text_train: pandas.DataFrame
        :param text_test: Pandas DataFrame containing the text test data
        :type text_test: pandas.DataFrame
        :param max_features: Maximum size for bag of words
        :type max_features: int
        :return: Pandas DataFrame containing training data and Pandas DataFrame containing test data
        :rtype: pandas.DataFrame
        """
    text_columns = ["skills", "title", "snippet"]
    for text_column_name in text_columns:
        vectorizer, train_data_features = prepare_text_train(text_train, text_column_name, max_features)
        test_data_features = prepare_text_test(text_test, text_column_name, vectorizer)

        column_names = ["$token_" + text_column_name + "_" + s for s in vectorizer.get_feature_names()]

        tokens_train = pd.DataFrame(train_data_features, columns=column_names, index=df_train.index)
        df_train = pd.concat([df_train, tokens_train], axis=1)
        # print df_train[0:3]
        # print "################################ \n\n"

        tokens_test = pd.DataFrame(test_data_features, columns=column_names, index=df_test.index)
        df_test = pd.concat([df_test, tokens_test], axis=1)
        # print df_test[0:3]
        # print "################################ \n\n"

    return df_train, df_test

def addTextTokensToWholeDF(df, text, max_features=500, vectorizers={}):
    """ Add tokenized text to data frame

        :param df: Pandas DataFrame containing the structured data
        :type df: pandas.DataFrame
        :param text: Pandas DataFrame containing the text data
        :type text: pandas.DataFrame
        :param max_features: Maximum size for bag of words
        :type max_features: int
        :param vectorizers: Vectorizers, if already given (optional)
        :type vectorizers: dict of sklearn.feature_extraction.text.CountVectorizer
        :return: Pandas DataFrame containing structured data and text tokens, as well as the vectorizers
        :rtype: pandas.DataFrame
        """
    text_columns = ["skills", "title", "snippet"]
    for text_column_name in text_columns:
        if vectorizers.has_key(text_column_name):
            vectorizer = vectorizers[text_column_name]
            data_features = prepare_text_test(text, text_column_name, vectorizer)
        else:
            vectorizer, data_features = prepare_text_train(text, text_column_name, max_features)
            vectorizers[text_column_name] = vectorizer

        column_names = ["$token_" + text_column_name + "_" + s for s in vectorizer.get_feature_names()]

        tokens = pd.DataFrame(data_features, columns=column_names, index=df.index)
        df = pd.concat([df, tokens], axis=1)

    return df, vectorizers
