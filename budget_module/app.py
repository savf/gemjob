# -*- coding: utf-8 -*-
# Data Module
# Request sample to be stored as JSON using:
# -> http://localhost:5000/update_data/
# Provide a 'sample_size', 'days_posted' or 'page_offset' in post request

import os
working_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
import copy
import logging
from logging.handlers import RotatingFileHandler
import time
import rethinkdb as rdb
from flask import Flask, request
from flask_restful import Resource, Api
import pandas as pd
# sys.path.insert(0, 'C:/Users/B/Documents/MasterProject/')
from dm_lib.dm_data_preparation import prepare_single_job
from dm_lib.dm_budgetModel import budget_model_production,\
    prepare_single_job_budget_model, predict
# sys.path.pop(0)

pd.set_option('chained_assignment',None) # turns off SettingWithCopyWarning
pd.set_option('display.max_columns', 200)


app = Flask(__name__)
api = Api(app)

RDB_HOST = 'localhost'
# RDB_HOST = 'database_module' # TODO: uncomment this again?
RDB_PORT = 28015
RDB_DB = 'datasets'
RDB_JOB_OPTIMIZED_TABLE = 'jobs_optimized'

TARGET_NAME = 'budget'

GLOBAL_VARIABLE = {}

class Predictions(Resource):

    def post(self):
        local_variable = copy.deepcopy(GLOBAL_VARIABLE)
        if "model" in local_variable:
            json_data = request.get_json(force=True)

            data_frame = prepare_single_job(json_data)
            unnormalized_data = data_frame.copy()

            if 'job_type' in unnormalized_data and \
                unnormalized_data['job_type'][0] != 'Hourly':
                normalized_data =\
                    prepare_single_job_budget_model(unnormalized_data,
                                                    local_variable['columns'],
                                                    local_variable['min'],
                                                    local_variable['max'],
                                                    local_variable['vectorizers'])

                prediction = predict(normalized_data, TARGET_NAME,
                                     local_variable['model'],
                                     local_variable['min'],
                                     local_variable['max'])

                app.logger.info("{} Prediction: {}".format(TARGET_NAME,
                                                           prediction))

                return {TARGET_NAME: prediction}

        return {}


api.add_resource(Predictions, '/get_predictions/')


class Update(Resource):

    def post(self):
        success = budget_model_built(max_tries=5)

        return {"success": success}


api.add_resource(Update, '/update_model/')


@app.route('/')
def start():
    try:
        local_variable = copy.deepcopy(GLOBAL_VARIABLE)
        if "model" in local_variable:
            model_info = len(local_variable["model"])
            return "<h1>Budget Module</h1><p>Model info: </p>" + str(model_info)
        else:
            return "<h1>Budget Module</h1><p>Not setup!</p>"
    except Exception as e:
        return "<h1>Cluster Module</h1><p>Never updated</p>"


def build_budget_model(connection):
    optimized_jobs_empty = rdb.db(RDB_DB).table(
        RDB_JOB_OPTIMIZED_TABLE).is_empty().run(connection)

    if not optimized_jobs_empty:
        model, columns, min, max, vectorizers =\
            budget_model_production(connection, budget_name=TARGET_NAME,
                                    normalization=True)

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


def budget_model_built(max_tries=-1):
    isSetup = False
    if max_tries > 0:
        n = 0
    else:
        n = max_tries-1
    while not isSetup and n < max_tries:
        connection = False

        try:
            connection = rdb.connect(RDB_HOST, RDB_PORT)

            if not rdb.db_list().contains(RDB_DB).run(connection):
                rdb.db_create(RDB_DB).run(connection)
            if not rdb.db(RDB_DB).table_list().contains(RDB_JOB_OPTIMIZED_TABLE).run(connection):
                rdb.db(RDB_DB).table_create(RDB_JOB_OPTIMIZED_TABLE).run(connection)

            isSetup = build_budget_model(connection)
        except Exception as e:
            app.logger.error('DB error:', e)
            if connection:
                connection.close()
            time.sleep(5)
            if max_tries > 0:
                n = n+1

    return isSetup


if __name__ == '__main__':
    app.logger.setLevel(logging.INFO)
    if budget_model_built():
        app.run(debug=True, use_debugger=False, use_reloader=False,
                host="0.0.0.0", port=5003)
