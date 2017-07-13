# -*- coding: utf-8 -*-
# Data Module
# Request sample to be stored as JSON using:
# -> http://localhost:5000/update_data/
# Provide a 'sample_size', 'days_posted' or 'page_offset' in post request

import os
working_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
import copy
import logging
import time
import rethinkdb as rdb
from flask import Flask, request
from flask_restful import Resource, Api
import pandas as pd
# sys.path.insert(0, 'C:/Users/B/Documents/MasterProject/')
from dm_lib.dm_data_preparation import prepare_single_job
from dm_lib.dm_jobTypeModel import prepare_single_job_jobtype_model, predict,\
    jobtype_model_production
from dm_lib.parameters import *
# sys.path.pop(0)

pd.set_option('chained_assignment', None) # turns off SettingWithCopyWarning
pd.set_option('display.max_columns', 200)

app = Flask(__name__)
api = Api(app)

TARGET_NAME = 'job_type'

GLOBAL_VARIABLE = {}


class Predictions(Resource):

    def post(self):
        local_variable = copy.deepcopy(GLOBAL_VARIABLE)
        if "model" in local_variable:
            json_data = request.get_json(force=True)

            data_frame = prepare_single_job(json_data)
            unnormalized_data = data_frame.copy()

            normalized_data =\
                prepare_single_job_jobtype_model(unnormalized_data,
                                                 TARGET_NAME,
                                                 local_variable['columns'],
                                                 local_variable['min'],
                                                 local_variable['max'],
                                                 local_variable['vectorizers'])

            prediction = predict(normalized_data, TARGET_NAME,
                                 local_variable['model'])

            app.logger.info("{} Prediction: {}".format(TARGET_NAME,
                                                       prediction))

            return {TARGET_NAME: prediction}

        return {}


api.add_resource(Predictions, '/get_predictions/')


class Update(Resource):

    def post(self):
        success = jobtype_model_built(max_tries=5)

        return {"success": success}


api.add_resource(Update, '/update_model/')


@app.route('/')
def start():
    try:
        local_variable = copy.deepcopy(GLOBAL_VARIABLE)
        if "model" in local_variable:
            model_name = local_variable["model"].__class__.__name__
            return "<h1>Job type Module</h1><p>Model used: </p>" + str(model_name)
        else:
            return "<h1>Job type Module</h1><p>Not setup!</p>"
    except Exception as e:
        return "<h1>Job type Module</h1><p>Never updated</p>"


def build_jobtype_model(connection):
    optimized_jobs_empty = rdb.db(RDB_DB).table(
        RDB_JOB_OPTIMIZED_TABLE).is_empty().run(connection)

    if not optimized_jobs_empty:
        model, columns, min, max, vectorizers =\
            jobtype_model_production(connection, normalization=True)

        local_variable = {}
        local_variable["model"] = model
        local_variable["columns"] = columns
        local_variable["min"] = min
        local_variable["max"] = max
        local_variable["vectorizers"] = vectorizers

        global GLOBAL_VARIABLE
        GLOBAL_VARIABLE = local_variable

        return True

    return False


def jobtype_model_built(max_tries=-1):
    is_setup = False
    if max_tries > 0:
        n = 0
    else:
        n = max_tries-1
    while not is_setup and n < max_tries:
        connection = False

        try:
            connection = rdb.connect(RDB_HOST, RDB_PORT)

            if not rdb.db_list().contains(RDB_DB).run(connection):
                rdb.db_create(RDB_DB).run(connection)
            if not rdb.db(RDB_DB).table_list().contains(RDB_JOB_OPTIMIZED_TABLE).run(connection):
                rdb.db(RDB_DB).table_create(RDB_JOB_OPTIMIZED_TABLE).run(connection)

            is_setup = build_jobtype_model(connection)
        except Exception as e:
            app.logger.error(e.message)
            if connection:
                connection.close()
            time.sleep(5)
            if max_tries > 0:
                n = n+1

    return is_setup


if __name__ == '__main__':
    app.logger.setLevel(logging.INFO)
    if jobtype_model_built():
        app.run(debug=True, use_debugger=False, use_reloader=False,
                host="0.0.0.0", port=5005)
