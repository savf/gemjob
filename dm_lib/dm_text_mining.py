from dm_general import evaluateClassification, evaluateRegression
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn import linear_model
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

def cleanText(df, text_column_name):
    isStr = isinstance(df[text_column_name].values[0], basestring)
    if isStr:
        stop_words = set(stopwords.words("english"))
        ps = PorterStemmer()
        df[text_column_name] = df.apply(lambda row: ' '.join(
            [ps.stem(w).lower() for w in word_tokenize(row[text_column_name]) if not w in stop_words]), axis=1)
    else:
        df[text_column_name] = df.apply(lambda row: ' '.join([w.lower() for w in row[text_column_name]]), axis=1)

    return df

def prepareTextTrain(df, text_column_name, max_features):
    df = cleanText(df, text_column_name)

    # bag of words
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,
                                 max_features=max_features)
    train_data_features = vectorizer.fit_transform(df[text_column_name])
    train_data_features = train_data_features.toarray()

    return vectorizer, train_data_features

def prepareTextTest(df, text_column_name, vectorizer):
    df = cleanText(df, text_column_name)

    # bag of words
    test_data_features = vectorizer.transform(df[text_column_name])
    test_data_features = test_data_features.toarray()

    return test_data_features

def textRegressionModel(df, label_name, train_data_features):
    # linear regression
    regr = svm.SVR(kernel='linear')#linear_model.LinearRegression()
    regr.fit(train_data_features, df[label_name])
    return regr

def textClassificationModel(df, label_name, train_data_features, n_estimators=100):
    # random forest classifier
    forest = RandomForestClassifier(n_estimators=n_estimators)
    forest.fit(train_data_features, df[label_name])
    return forest

def doTextMining(df_train, df_test, label_name, regression, max_features=5000):
    print "##############################\n    Text Mining for " + label_name + \
          "    \n##############################\n"

    text_columns=["skills", "title", "snippet"]
    for text_column_name in text_columns:
        print "### Predict ", label_name, " with: ", text_column_name, " ###"

        vectorizer, train_data_features = prepareTextTrain(df_train, text_column_name, max_features)
        test_data_features = prepareTextTest(df_test, text_column_name, vectorizer)
        if regression:
            model = textRegressionModel(df_train, label_name, train_data_features)
        else:
            model = textClassificationModel(df_train, label_name, train_data_features)

        predictions = model.predict(test_data_features)

        # evaluate
        if regression:
            evaluateRegression(df_test, predictions, label_name)
        else:
            evaluateClassification(df_test, predictions, label_name)

        print "### Predictions: ###"
        print predictions[0:8]
        print "### Actual values: ###"
        print df_test[label_name][0:8]
        print "###########"

    print "################################ \n\n"
    return predictions