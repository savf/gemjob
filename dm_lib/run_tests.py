from dm_data_exploration import explore_data
from dm_data_preparation import *
from dm_text_mining import *
from dm_budgetModel import budget_model
from parameters import *

pd.set_option('chained_assignment', None) # turns off SettingWithCopyWarning
pd.set_option('display.max_columns', 200)

def test_text_mining():
    """ Perform a sentiment analysis as a demonstration
    """
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

    vectorizer, train_data_features = prepare_text_train(df_train, text_column_name, max_features)
    test_data_features = prepare_text_test(df_test, text_column_name, vectorizer)

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

    model = text_classification_model(df_train, label_name, train_data_features)

    print "\n## Model: ##\n", model

    predictions = model.predict(test_data_features)

    print "\n## Predictions: ##\n", predictions
    print "\n## Actual Sentiment: ##\n", df_test[label_name].to_dict().values()

    evaluate_classification(df_test, predictions, label_name)

    print "##############################"


#run
db_setup("data/found_jobs_4K_extended.json", host=RDB_HOST)

connection = rdb.connect(RDB_HOST, RDB_PORT)
try:
    # prepare_data("data/found_jobs_4K_extended.json")
    budget_model("data/found_jobs_4K_extended.json", connection)
    # jobtype_model("data/found_jobs_4K_extended.json")
    # feedback_model("data/found_jobs_4K_extended.json")
    # experience_level_model("data/found_jobs_4K_extended.json")
    # test_text_mining()
    # explore_data("data/found_jobs_4K_extended.json")
    # test_clustering("data/found_jobs_4K_extended.json", "Mean-Shift")
    # test_knn("data/found_jobs_4K_extended.json")
except RqlRuntimeError as e:
    print 'Database error: {}'.format(e)
finally:
    connection.close()

