# -*- coding: utf-8 -*-
# Data Module
# Request sample to be stored as JSON using:
# -> http://localhost:5000/update_data/
# Provide a 'sample_size', 'days_posted' or 'page_offset' in post request

import json
import os
working_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
os.environ['HTTPLIB_CA_CERTS_PATH'] = working_dir + 'cacert.pem'

from time import strftime, gmtime

import sys
import rethinkdb as rdb
import upwork
from dateutil import tz
from flask import Flask, g, abort, request
from flask_restful import Resource, Api
from rethinkdb.errors import RqlRuntimeError, RqlDriverError

import credentials


app = Flask(__name__)
api = Api(app)

# RDB_HOST = 'localhost'
RDB_HOST = 'database_module'
RDB_PORT = 28015
RDB_DB = 'datasets'
RDB_TABLE = 'jobs'

max_tries = 10
max_request_size = 99


def db_setup():
    connection = rdb.connect(RDB_HOST, RDB_PORT)
    try:
        if not rdb.db_list().contains(RDB_DB).run(connection):
            rdb.db_create(RDB_DB).run(connection)
        if not rdb.db(RDB_DB).table_list().contains(RDB_TABLE).run(connection):
            rdb.db(RDB_DB).table_create(RDB_TABLE).run(connection)
    except RqlRuntimeError:
        print 'Database {} and table {} already exist.'.format(RDB_DB, RDB_TABLE)
    finally:
        connection.close()

###### class begin
class DataUpdater(Resource):  # Our class "DataUpdater" inherits from "Resource"

    # Print iterations progress
    def print_progress(self, iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            bar_length  - Optional  : character length of bar (Int)
        """
        str_format = "{0:." + str(decimals) + "f}"
        percents = str_format.format(100 * (iteration / float(total)))
        filled_length = int(round(bar_length * iteration / float(total)))
        bar = '█' * filled_length + '-' * (bar_length - filled_length)

        sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

        if iteration == total:
            sys.stdout.write('\n')
        sys.stdout.flush()

    ### post request
    def post(self):
        json_data = request.get_json(force=True)
        sample_size = 10
        days_posted = 365
        page_offset = None

        if 'sample_size' in json_data:
            sample_size = json_data['sample_size']
            if sample_size < 1:
                return {'api_name': 'Data module REST API', 'success': False, 'sample-size': 0,
                        'exception': 'sample_size too small'}
        if 'days_posted' in json_data:
            days_posted = json_data['days_posted']
            if days_posted < 1:
                return {'api_name': 'Data module REST API', 'success': False, 'sample-size': sample_size,
                        'exception': 'Only non-zero and positive values for days posted allowed'}
        if 'page_offset' in json_data:
            page_offset = json_data['page_offset']
            if page_offset == "None":
                page_offset = None

        found_jobs = []
        pages = 1 + (sample_size - 1) / max_request_size
        print 'pages: ' + str(pages)
        _sample_size = max_request_size

        exception = "None"

        # assemble data in multiple iterations because of maximum number of data we can request
        for p in range(0, pages):

            if p == pages - 1:
                _sample_size = (sample_size % max_request_size) if (sample_size % max_request_size) != 0 else sample_size
            # print 'paging: ' + str(p * max_request_size) + ';' + str(_sample_size)

            # connect to Upwork
            client = upwork.Client(public_key=credentials.public_key, secret_key=credentials.secret_key,
                                   oauth_access_token=credentials.oauth_access_token,
                                   oauth_access_token_secret=credentials.oauth_access_token_secret,
                                   timeout=30)

            query_data = {'q': '*', 'category2': 'Data Science & Analytics',
                          'job_status': 'completed', 'days_posted': days_posted}

            # try to get data until we either got it or we exceed the limit
            # for i in range(0, max_tries):
            #     try:
            #         if page_offset is None:
            #             found_jobs.extend(
            #                 client.provider_v2.search_jobs(data=query_data, page_offset=(p * max_request_size),
            #                                                page_size=_sample_size))
            #         else:
            #             found_jobs.extend(
            #                 client.provider_v2.search_jobs(data=query_data,
            #                                                page_offset=page_offset + (p * max_request_size),
            #                                                page_size=_sample_size))
            #         print 'Successfully found jobs, page_offset=' + str(p * max_request_size) + ', page_size=' + str(
            #             _sample_size)
            #         exception = "None"
            #         break
            #     except Exception as e:
            #         print 'Number of tries for job search: ' + str(i)
            #         print e
            #         exception = str(e.code) + ' - ' + e.msg
        with open(working_dir + 'found_jobs_4K.json') as data_file:
            found_jobs = json.load(data_file)

        if found_jobs is not None:

            counter = 0
            for job in found_jobs:
                # Save already found profiles in 10% progress steps
                if counter % int(round(len(found_jobs)/10)) == 0:
                    try:
                        with open(working_dir + strftime("found_jobs_%d.%m.-%H%M.json", gmtime()), "a+") as f:
                            f.truncate()
                            f.write(json.dumps(found_jobs))
                    except Exception as e:
                        print '\r\n Problems writing to JSON file, aborted.\r\n'

                for i in range(0, max_tries):
                    try:
                        job_profile = client.job.get_job_profile(job["id"].encode('UTF-8'))
                        counter += 1
                        self.print_progress(counter, len(found_jobs), prefix = 'Progress:', suffix = 'Complete', bar_length = 50)
                        assignment_info = job_profile['assignment_info']['info']
                        # For jobs with only a single freelancer, assignment_info directly contains the info
                        if not isinstance(assignment_info, list):
                            assignment_info = [assignment_info]
                        # Create the divisor to build the average of the individual feedback elements
                        divisor = float(len(assignment_info))
                        for info in assignment_info:
                            if 'feedback_for_buyer' not in info and 'feedback_for_provider' not in info:
                                divisor -= 1.0

                        total_charge = 0
                        for info in assignment_info:
                            total_charge += float(info['total_charge'])
                            if 'feedback_for_provider' in info:
                                for feedback in info['feedback_for_provider']['scores']['score']:
                                    if 'feedback_for_freelancer_{}'.format(feedback['label'].lower()) in job:
                                        job['feedback_for_freelancer_{}'.format(feedback['label'].lower())] +=\
                                            float(feedback['score']) / divisor
                                    else:
                                        job['feedback_for_freelancer_{}'.format(feedback['label'].lower())] = \
                                            float(feedback['score']) / divisor
                            if 'feedback_for_buyer' in info:
                                for feedback in info['feedback_for_buyer']['scores']['score']:
                                    if 'feedback_for_client_{}'.format(feedback['label'].lower()) in job:
                                        job['feedback_for_client_{}'.format(feedback['label'].lower())] +=\
                                            float(feedback['score']) / divisor
                                    else:
                                        job['feedback_for_client_{}'.format(feedback['label'].lower())] = \
                                            float(feedback['score']) / divisor

                        job['freelancer_count'] = unicode(len(assignment_info))
                        job['total_charge'] = unicode(total_charge)
                        break
                    except Exception as e:
                        if hasattr(e, 'code') and e.code == 403:
                            print '\r\n Profile access denied for job {} \r\n'.format(job['id'])
                            break
                        print 'Number of tries for job profile: {}'.format(str(i))
                        print e

            with open(working_dir + strftime("found_jobs_%d.%m.-%H%M.json", gmtime()), "a+") as f:
                f.truncate()
                f.write(json.dumps(found_jobs))

            response = rdb.table(RDB_TABLE).insert(found_jobs, conflict="replace").run(g.rdb_conn)

            success_criterion = len(found_jobs) == sample_size and response['inserted'] > 0
            if len(found_jobs) != sample_size:
                exception = "Only got {} of the requested {} jobs".format(len(found_jobs), sample_size)
            if response['inserted'] <= 0:
                exception = "No new samples were returned by the API"
            return {'api_name': 'Data module REST API',
                    'success': success_criterion,
                    'sample-size': len(found_jobs),
                    'exception': exception,
                    'database-response': response}

        return {'api_name': 'Data module REST API', 'success': False,
                'sample-size': 0, 'exception': exception}

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


api.add_resource(DataUpdater, '/update_data/')


@app.route('/')
def start():
    try:
        last_updated = rdb.table(RDB_TABLE).order_by("requested_on").get_field("requested_on").max().run(g.rdb_conn)
        target_zone = tz.gettz('Europe/Zurich')
        last_updated = last_updated.replace(tzinfo=tz.gettz('UTC'))
        last_updated = last_updated.astimezone(target_zone)
        return "<h1>Data Module</h1><p>Last updated: {} </p>".format(last_updated)
    except Exception as e:
        return "<h1>Data Module</h1><p>Never updated</p>"


if __name__ == '__main__':
    db_setup()
    app.run(debug=True, use_debugger=False, use_reloader=False, host="0.0.0.0")
