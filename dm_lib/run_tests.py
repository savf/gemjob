from dm_data_preparation import *
from dm_text_mining import *
from dm_jobTypeModel import jobTypeModel
from dm_budgetModel import budgetModel
import pandas as pd
import numpy as np

pd.set_option('chained_assignment',None) # turns off SettingWithCopyWarning

def testTextMining():
    print "\n\n### Testing Text Mining ###\n"
    max_features = 5000
    text_column_name = 'sentences'
    label_name = "sentiment"
    df_train = pd.DataFrame({text_column_name: ['This is a very good site. I will recommend it to others.', 'Aweful page, I hate it',
                                                'good work! keep it up', 'Terrible site, seriously aweful'],
                       label_name: ["pos", "neg", "pos", "neg"]})

    df_test = pd.DataFrame({text_column_name: ['This page is soo good!', 'This is really really good',
                                               'Your layout is seriously aweful', 'The most terrible site on the interwebs'],
                             label_name: ["pos", "pos", "neg", "neg"]})

    vectorizer, train_data_features = prepareTextTrain(df_train, text_column_name, max_features)
    test_data_features = prepareTextTest(df_test, text_column_name, vectorizer)

    print "\n## DataFrame: ##\n", df_train
    print "\n## Shape of Features: ##\n", train_data_features.shape
    print "\n## Features: ##\n", train_data_features
    vocab = vectorizer.get_feature_names()
    print "\n## Vocab: ##\n", vocab

    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data_features, axis=0)
    # For each, print the vocabulary word and the number of times it
    # appears in the training set
    print "\n## Frequency of words: ##"
    for tag, count in zip(vocab, dist):
        print count, tag

    model = textClassificationModel(df_train,label_name, train_data_features)

    print "\n## Model: ##\n", model

    predictions = model.predict(test_data_features)

    print "\n## Predictions: ##\n", predictions
    print "\n## Actual Sentiment: ##\n", df_test[label_name].to_dict().values()

    evaluateClassification(df_test, predictions, label_name)

    print "##############################"

#run
budgetModel("data/found_jobs_4K_extended.json")
# jobTypeModel("data/found_jobs_4K_extended.json")
# testTextMining()