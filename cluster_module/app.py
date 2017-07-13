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
from dm_data_preparation import prepare_data, load_data_frame_from_db, prepare_single_job
from dm_clustering import do_clustering_mean_shift, prepare_single_job_clustering, predict
# sys.path.pop(0)
import copy

pd.set_option('chained_assignment',None) # turns off SettingWithCopyWarning
pd.set_option('display.max_columns', 200)


app = Flask(__name__)
api = Api(app)

RDB_HOST = 'localhost'
# RDB_HOST = 'database_module' # TODO: uncomment this again?
RDB_PORT = 28015
RDB_DB = 'datasets'
RDB_JOB_OPTIMIZED_TABLE = 'jobs_optimized'


GLOBAL_VARIABLE = {}

class Predictions(Resource):

    def post(self):
        local_variable = copy.deepcopy(GLOBAL_VARIABLE)
        if local_variable.has_key("clusters"):
            json_data = request.get_json(force=True)

            data_frame = prepare_single_job(json_data)
            unnormalized_data = data_frame.copy()

            normalized_data, target_columns = prepare_single_job_clustering(data_frame, local_variable["centroids"].columns, local_variable["min"], local_variable["max"], vectorizers=local_variable["vectorizers"], weighting=True)

            unnormalized_data = predict(unnormalized_data, normalized_data, local_variable["clusters"], local_variable["centroids"], target_columns)

            predictions = unnormalized_data.to_dict('records')

            print "\n\n### Predictions:"
            print predictions

            return predictions[0]
        else:
            return {}


api.add_resource(Predictions, '/get_predictions/')


class Update(Resource):

    def post(self):
        success = clustering_setup(max_tries=5)

        return {"success": success}


api.add_resource(Update, '/update_clusters/')


@app.route('/')
def start():
    local_variable = copy.deepcopy(GLOBAL_VARIABLE)
    try:
        if local_variable.has_key("clusters"):
            number_of_clusters = len(local_variable["clusters"])
            return "<h1>Cluster Module</h1><p>Number of clusters: </p>" + str(number_of_clusters)
        else:
            return "<h1>Cluster Module</h1><p>Not setup!</p>"
    except Exception as e:
        return "<h1>Cluster Module</h1><p>Error</p>"


def cluster_data(connection):
    if connection is not None:
        data_in_db = not rdb.db(RDB_DB).table(RDB_JOB_OPTIMIZED_TABLE).is_empty().run(connection)
    else:
        data_in_db = True

    if data_in_db:
        # load data
        if connection is None:
            data_frame = prepare_data("data/found_jobs_4K_extended.json")
        else:
            data_frame = load_data_frame_from_db(connection=connection)
        print "# number of jobs in db:", data_frame.shape[0]

        # cluster using mean shift
        _, clusters, centroids, min, max, vectorizers = do_clustering_mean_shift(data_frame, find_best_params=False, do_explore=False, min_rows_per_cluster=3)
        print "# new clusters computed"

        # store everything as global variable
        local_variable = {}
        local_variable["clusters"] = clusters
        local_variable["centroids"] = centroids
        local_variable["min"] = min
        local_variable["max"] = max
        local_variable["vectorizers"] = vectorizers
        # only one atomic write! -> no conflicts
        global GLOBAL_VARIABLE
        GLOBAL_VARIABLE = local_variable
        print "# new clusters stored"

        return True

    return False


def clustering_setup(max_tries=-1):
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

            isSetup = cluster_data(connection)
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
        isSetup = cluster_data(None)

    return isSetup


if __name__ == '__main__' and clustering_setup(3):
    app.run(debug=True, use_debugger=False, use_reloader=False, host="0.0.0.0", port=5002)
