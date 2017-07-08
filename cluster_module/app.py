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
# sys.path.insert(0, 'C:/Users/B/Documents/MasterProject/dm_lib/')
# print(sys.path)
from dm_data_preparation import prepare_data, load_data_frame_from_db
from dm_clustering import do_clustering_mean_shift
# sys.path.pop(0)
import thread

app = Flask(__name__)
api = Api(app)

RDB_HOST = 'localhost'
# RDB_HOST = 'cluster_module' # TODO: uncomment this again?
RDB_PORT = 28015
RDB_DB = 'datasets'
RDB_JOB_OPTIMIZED_TABLE = 'jobs_optimized'
# RDB_CLUSTER_TABLE = 'clusters'
# RDB_CLUSTER_CENTROIDS_TABLE = 'clusters_centroids'
# RDB_CLUSTER_MIN_TABLE = 'clusters_min'
# RDB_CLUSTER_MAX_TABLE = 'clusters_max'
# RDB_CLUSTER_VEC_TABLE = 'clusters_vectorizers'

g_clusters = None
g_centroids = None
g_min = None
g_max = None
g_vectorizers = None

###### class begin
class Predictions(Resource):

    ### post request
    def post(self):
        json_data = request.get_json(force=True)

        possible_request_columns = ["subcategory2", "title", "description", "freelancer_count", "skills", "start_date",
                                    "job_type", "budget", "duration", "workload", "experience_level"]

        # TODO: change names of columns to be consistent with prepare_data
        # TODO: prepare data like in dm_lib, but without removing missing columns
        # TODO: identify target_columns (like total_charge and form fields that were not filled
        # TODO: get clusters from data_base
        # TODO: predict


    ### post end


###### class end

@app.before_request
def before_request():
    try:
        g.rdb_conn = rdb.connect(host=RDB_HOST, port=RDB_PORT, db=RDB_DB)
    except RqlDriverError:
        abort(503, "No database connection could be established.")


@app.teardown_request
def teardown_request(exception):
    try:
        g.rdb_conn.close()
    except AttributeError:
        pass


api.add_resource(Predictions, '/get_predictions/')


@app.route('/')
def start():
    try:
        # number_of_clusters = rdb.table(RDB_CLUSTER_TABLE).count().run(g.rdb_conn)
        number_of_clusters = len(g_clusters)
        return "<h1>Cluster Module</h1><p>Number of clusters: </p>" + str(number_of_clusters)
    except Exception as e:
        return "<h1>Cluster Module</h1><p>Never updated</p>"


def cluster_data(connection):
    data_in_db = not rdb.db(RDB_DB).table(RDB_JOB_OPTIMIZED_TABLE).is_empty().run(connection)

    if data_in_db:
        not_stored = True
        max_num_tries = 5
        # load data
        # data_frame = prepare_data("data/found_jobs_4K_extended.json") # for testing only!
        data_frame = load_data_frame_from_db(connection=connection)

        # cluster using mean shift
        _, clusters, centroids, min, max, vectorizers = do_clustering_mean_shift(data_frame, find_best_params=False, do_explore=False)
        print "# new clusters computed"

        # store everything as global variable
        global g_clusters
        global g_centroids
        global g_min
        global g_max
        global g_vectorizers
        g_clusters = clusters
        g_centroids = centroids
        g_min = min
        g_max = max
        g_vectorizers = vectorizers
        print "# new clusters stored"

        return True

        # # store in DB: indices with cluster IDs -> (can use this to find clusters in unnormalized data when predicting)
        # data_frame['id'] = data_frame.index
        # data_frame = data_frame[['cluster_label', 'id']] # IMPORTANT: if only selecting one column "data_frame['c']", ".to_dict('records')" doesn't work anymore

        # num_tries = 0
        # while num_tries < max_num_tries and not_stored:
        #     try:
        #         rdb.db(RDB_DB).table(RDB_CLUSTER_TABLE).insert(data_frame.to_dict('records'), conflict="replace").run(connection)
        #
        #         # store in DB: centroids, min, max, vectorizers
        #         rdb.db(RDB_DB).table(RDB_CLUSTER_CENTROIDS_TABLE).insert(centroids.to_dict('records'), conflict="replace").run(connection)
        #         rdb.db(RDB_DB).table(RDB_CLUSTER_MIN_TABLE).insert(min.to_dict('records'), conflict="replace").run(connection)
        #         rdb.db(RDB_DB).table(RDB_CLUSTER_MAX_TABLE).insert(max.to_dict('records'), conflict="replace").run(connection)
        #
        #         rdb.db(RDB_DB).table(RDB_CLUSTER_VEC_TABLE).insert(vectorizers.to_dict('records'), conflict="replace").run(connection)
        #     except Exception as e:
        #         print 'DB error:', e
        #         num_tries = num_tries+1
        #         time.sleep(3)
        # if not_stored:
        #     raise Exception, "Cannot store clusters"
    return False

def subscribe_db():
    notSetup = True

    while (notSetup):
        connection = False

        try:
            connection = rdb.connect(RDB_HOST, RDB_PORT)
            feed = rdb.db(RDB_DB).table(RDB_JOB_OPTIMIZED_TABLE).changes({"squash": True}).run(connection) # IMPORTANT: set squash to true to only get notified of a change once
            for change in feed:
                print "\n\n### Change in DB occured\n"
                print "#Change:    ",change, "\n"
                # cluster_data(connection) # TODO activate this again
            notSetup = False
        except Exception as e:
            print 'DB error:', e
            if connection:
                connection.close()
            time.sleep(5)

def clustering_setup():
    notSetup = True

    while(notSetup):
        connection = False

        try:
            connection = rdb.connect(RDB_HOST, RDB_PORT)

            if not rdb.db_list().contains(RDB_DB).run(connection):
                rdb.db_create(RDB_DB).run(connection)
            if not rdb.db(RDB_DB).table_list().contains(RDB_JOB_OPTIMIZED_TABLE).run(connection):
                rdb.db(RDB_DB).table_create(RDB_JOB_OPTIMIZED_TABLE).run(connection)
            # if not rdb.db(RDB_DB).table_list().contains(RDB_CLUSTER_TABLE).run(connection):
            #     rdb.db(RDB_DB).table_create(RDB_CLUSTER_TABLE).run(connection)

            # cluster_data(connection) # TODO activate this again

            # subscribe to changes in DB
            thread.start_new_thread(subscribe_db, ())

            notSetup = False
        except Exception as e:
            print 'DB error:', e
            if connection:
                connection.close()
            time.sleep(5)


if __name__ == '__main__':
    with app.app_context():
        clustering_setup() # TODO Do this in another thread!
        app.run(debug=True, use_debugger=False, use_reloader=False, host="0.0.0.0")
