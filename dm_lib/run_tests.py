# from dm_data_exploration import explore_data
from dm_data_preparation import *
# from dm_lib.dm_jobTypeModel import jobtype_model_production, \
#     jobtype_model_development
from dm_text_mining import *
from dm_budgetModel import budget_model_development, budget_model_production
from dm_feedbackModel import feedback_model_development, \
    feedback_model_production
from parameters import *
from dm_clustering import test_clustering
from dm_knn import test_knn
from dm_general import evaluate_predictions_regression, evaluate_predictions_classification

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


RDB_HOST = "localhost"

#run

# exp_var_sc, abs_err, sq_err, mape = evaluate_predictions_regression(test_clustering, (JOBS_FILE, "Mean-Shift", "budget"), 5)
# exp_var_sc, abs_err, sq_err, mape = evaluate_predictions_regression(test_knn, (JOBS_FILE, "budget"), 15)
accuracy = evaluate_predictions_classification(test_clustering, (JOBS_FILE, "Mean-Shift", "job_type"), 2)

# test_clustering(JOBS_FILE, "Mean-Shift", target="budget")
# test_knn(JOBS_FILE, target="budget")
# db_setup(JOBS_FILE, host=RDB_HOST)
#
# connection = rdb.connect(RDB_HOST, RDB_PORT)
# try:
#     # prepare_data(JOBS_FILE)
#     budget_model_development(JOBS_FILE, connection)
#     # budget_model_production(connection)
#     # jobtype_model_production(connection)
#     # jobtype_model_development(JOBS_FILE, connection)
#     # feedback_model_development(JOBS_FILE, connection)
#     # feedback_model_production(connection, normalization=False)
#     # experience_level_model(JOBS_FILE)
#     # test_text_mining()
#     # explore_data(JOBS_FILE)
#     # test_knn(JOBS_FILE)
# except RqlRuntimeError as e:
#     print 'Database error: {}'.format(e)
# finally:
#     connection.close()
