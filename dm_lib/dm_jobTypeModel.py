from dm_data_preparation import *
from dm_text_mining import doTextMining

def prepareDataJobTypeModel(data_frame):
    # TODO
    ### convert everything to nominal
    convertToNominal(data_frame)
    print "prepareDataJobTypeModel NOT IMPLEMENTED"
    return data_frame

def jobTypeModel(file_name):
    label_name = "job_type"
    data_frame = prepareData(file_name)
    data_frame = prepareDataJobTypeModel(data_frame)

    df_train, df_test = splitIntoTestTrainSet(data_frame, 0.8)

    doTextMining(df_train, df_test, label_name, regression=False, max_features=5000)

    # printCorr(data_frame, label_name)