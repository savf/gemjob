# -*- coding: utf-8 -*-
# Data Module
# Request sample to be stored as JSON using:
# -> http://localhost:5000/update_data/
# Provide a 'sample_size', 'days_posted' or 'page_offset' in post request

import json
import os
working_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
import sys
import time
import rethinkdb as rdb
from flask import Flask, g, abort, request
from flask_restful import Resource, Api
from rethinkdb.errors import RqlRuntimeError, RqlDriverError
import pandas as pd
# sys.path.insert(0, 'C:/Users/B/Documents/MasterProject/')
from dm_lib.dm_data_preparation import prepare_data, load_data_frame_from_db, prepare_single_job
from dm_lib.dm_clustering import prepare_single_job_clustering, prepare_data_clustering
from dm_lib.dm_knn import predict_knn
from dm_lib.parameters import *
# sys.path.pop(0)
import copy

pd.set_option('chained_assignment',None) # turns off SettingWithCopyWarning
pd.set_option('display.max_columns', 200)


app = Flask(__name__)
api = Api(app)

K = 15
GLOBAL_VARIABLE = {}

def get_knn_prediction(json_data, local_variable, target="text"):
    if target == "text":
        get_example_text = True
    else:
        get_example_text = False

    data_frame = prepare_single_job(json_data)
    unnormalized_data = data_frame.copy()

    normalized_data, target_columns = prepare_single_job_clustering(data_frame,
                                                                    local_variable["data_frame_normalized"].columns,
                                                                    local_variable["min"], local_variable["max"],
                                                                    vectorizers=local_variable["vectorizers"],
                                                                    weighting=True, do_log_transform=True)

    if target == "budget":
        target_columns.append(target)

    # Reweighting harms budget! Benefits subcategory however and shows more similar example snippet and title
    predictions = predict_knn(unnormalized_data, local_variable["data_frame_original"], normalized_data,
                              local_variable["data_frame_normalized"], k=K, target_columns=target_columns,
                              do_reweighting=get_example_text, get_example_text=get_example_text)

    predictions = predictions.to_dict('records')

    print "\n\n### Predictions:"
    print predictions

    return predictions[0]


class PredictionsText(Resource):

    def post(self):
        local_variable = copy.deepcopy(GLOBAL_VARIABLE)
        if local_variable.has_key("data_frame_normalized"):
            json_data = request.get_json(force=True)
            return get_knn_prediction(json_data, local_variable, target="text")
        else:
            return {}


api.add_resource(PredictionsText, '/get_predictions/text')


class PredictionsBudget(Resource):

    def post(self):
        local_variable = copy.deepcopy(GLOBAL_VARIABLE)
        if local_variable.has_key("data_frame_normalized"):
            json_data = request.get_json(force=True)
            return get_knn_prediction(json_data, local_variable, target="budget")
        else:
            return {}


api.add_resource(PredictionsBudget, '/get_predictions/budget')


class Update(Resource):

    def post(self):
        success = knn_setup(max_tries=5)

        return {"success": success}


api.add_resource(Update, '/update_knn/', '/update_model/')


@app.route('/')
def start():
    local_variable = copy.deepcopy(GLOBAL_VARIABLE)
    try:
        if local_variable.has_key("data_frame_normalized"):
            number_of_jobs = local_variable["data_frame_normalized"].shape[0]
            return "<h1>kNN Module</h1><p>Number of jobs stored: </p>" + str(number_of_jobs) + " <p>k: " + str(K) + "</p>"
        else:
            return "<h1>kNN Module</h1><p>Not setup!</p>"
    except Exception as e:
        return "<h1>kNN Module</h1><p>Error</p>"


def update_knn(connection):
    try:
        # load data
        data_frame = load_data_frame_from_db(connection=connection)
        data_frame_original = data_frame.copy()
        print "# number of jobs in db:", data_frame.shape[0]

        # prepare for knn
        data_frame_normalized, min, max, vectorizers = prepare_data_clustering(data_frame, z_score_norm=False, add_text=True, weighting=True, do_log_transform=True)
        print "# data prepared"

        # store everything as global variable
        local_variable = {}
        local_variable["data_frame_normalized"] = data_frame_normalized
        local_variable["data_frame_original"] = data_frame_original
        local_variable["min"] = min
        local_variable["max"] = max
        local_variable["vectorizers"] = vectorizers
        # only one atomic write! -> no conflicts
        global GLOBAL_VARIABLE
        GLOBAL_VARIABLE = local_variable
        print "# kNN module updated"

        return True
    except Exception as e:
        print 'DB error:', e
        return False


def knn_setup(max_tries=-1):
    isSetup = False
    if max_tries > 0:
        n = 0
    else:
        n = max_tries-1
    while(not isSetup and n < max_tries):
        connection = False

        try:
            connection = rdb.connect(RDB_HOST, RDB_PORT)

            if not rdb.db_list().contains(RDB_DB).run(connection):
                rdb.db_create(RDB_DB).run(connection)
            if not rdb.db(RDB_DB).table_list().contains(RDB_JOB_OPTIMIZED_TABLE).run(connection):
                rdb.db(RDB_DB).table_create(RDB_JOB_OPTIMIZED_TABLE).run(connection)

            isSetup = update_knn(connection)
        except Exception as e:
            print 'DB error:', e
            if connection:
                connection.close()
            time.sleep(5)
            if max_tries > 0:
                n = n+1

    if not isSetup:
        # if still not setup: use data file
        print "# No DB connection: Using backup file"
        isSetup = update_knn(None)

    return isSetup


if __name__ == '__main__' and knn_setup(3):
    app.run(debug=True, use_debugger=False, use_reloader=False, host="0.0.0.0", port=5006)
