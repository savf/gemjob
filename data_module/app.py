# -*- coding: utf-8 -*-
# Data Module
# Request sample to be stored as JSON using:
# -> http://localhost:5000/update_data/
# Provide a 'sample_size', 'days_posted' or 'page_offset' in post request

import json
import os

from dm_lib.dm_data_preparation import prepare_data
from dm_lib.parameters import *

working_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
os.environ['HTTPLIB_CA_CERTS_PATH'] = working_dir + 'cacert.pem'

from time import strftime, gmtime

import sys
import grequests
import rethinkdb as rdb
import upwork
from dateutil import tz
import requests
from flask import Flask, g, abort, request
from flask_restful import Resource, Api
from rethinkdb.errors import RqlRuntimeError, RqlDriverError
from datetime import datetime
from dm_lib.parameters import *

import credentials


app = Flask(__name__)
api = Api(app)

max_tries = 10
max_request_size = 99


def db_setup():
    connection = rdb.connect(RDB_HOST, RDB_PORT)
    try:
        if not rdb.db_list().contains(RDB_DB).run(connection):
            rdb.db_create(RDB_DB).run(connection)
        if not rdb.db(RDB_DB).table_list().contains(RDB_JOB_TABLE).run(connection):
            rdb.db(RDB_DB).table_create(RDB_JOB_TABLE).run(connection)
        if not rdb.db(RDB_DB).table_list().contains(RDB_PROFILE_TABLE).run(connection):
            rdb.db(RDB_DB).table_create(RDB_PROFILE_TABLE).run(connection)
    except RqlRuntimeError:
        print 'Database {} and table {} already exist.'.format(RDB_DB, RDB_JOB_TABLE)
    finally:
        connection.close()


###### class begin
class DataUpdater(Resource):  # Our class "DataUpdater" inherits from "Resource"

    mining_module_urls = {'CL': 'http://cluster_module:5002/',
                          'BU': 'http://budget_module:5003/',
                          'FE': 'http://feedback_module:5004/',
                          'JO': 'http://jobtype_module:5005/'}

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

    @staticmethod
    def median(lst):
        lst = sorted(lst)
        if len(lst) < 1:
            return None
        if len(lst) % 2 == 1:
            return lst[((len(lst) + 1) / 2) - 1]
        else:
            return float(sum(lst[(len(lst) / 2) - 1:(len(lst) / 2) + 1])) / 2.0

    @staticmethod
    def is_int(s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    @staticmethod
    def is_float(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    @staticmethod
    def safe_modulo(first, second):
        if second == 0:
            return 0
        return first % second

    @staticmethod
    def clamp(value, lower, upper):
        return max(lower, min(value, upper))

    def enrich_with_job_profiles(self, client, found_jobs,
                                 save_incrementally=False, stored_job_profiles=None):
        """ Add additional features to a given list of jobs obtained from Upwork

        :param client: Upwork Client to access the API
        :param found_jobs: List with jobs obtained from Upwork
        :param save_incrementally: Whether to save in 10% steps
        :param stored_job_profiles: Job profiles already requested from Upwork
        :return:
        """
        job_profiles = []
        counter = 0
        for job in found_jobs:
            # Save already found profiles in 10% progress steps
            if save_incrementally and self.safe_modulo(counter, int(round(len(found_jobs) / 10))) == 0:
                try:
                    with open(working_dir + strftime("found_jobs_%d.%m.-%H%M.json", gmtime()), "a+") as f:
                        f.truncate()
                        f.write(json.dumps(found_jobs))
                    with open(working_dir + strftime("job_profiles_%d.%m.-%H%M.json", gmtime()), "a+") as f:
                        f.truncate()
                        f.write(json.dumps(job_profiles))
                except Exception as e:
                    print '\r\n Problems writing to JSON file, aborted.\r\n'
            try:
                del job_profile
            except:
                pass

            if stored_job_profiles is None:
                for i in range(0, max_tries):
                    try:
                        job_profile = client.job.get_job_profile(job["id"].encode('UTF-8'))
                        if job_profile['ciphertext'] == job['id']:
                            # Replace the key 'ciphertext' with 'id' to make it easier to use with RethinkDB later
                            job_profile.pop('ciphertext')
                            job_profile['id'] = job['id']
                        break
                    except Exception as e:
                        if hasattr(e, 'code') and e.code == 403:
                            print '\r\n Profile access denied for job {} \r\n'.format(job['id'])
                            break
                        print 'Number of tries for job profile: {}'.format(
                            str(i))
                        print e
            else:
                if job['id'] in stored_job_profiles:
                    job_profile = stored_job_profiles[job['id']]
            try:
                job_profiles.append(job_profile)
                counter += 1
                self.print_progress(counter, len(found_jobs),
                                    prefix='Progress:',
                                    suffix='Complete', bar_length=50)
                assignment_info = job_profile['assignment_info']['info']
                # For jobs with only a single freelancer, assignment_info directly contains the info
                if not isinstance(assignment_info, list):
                    assignment_info = [assignment_info]
                # Create the divisors to build the average of the individual feedback elements
                divisor_client = 0.0
                divisor_freelancer = 0.0
                for info in assignment_info:
                    if 'feedback_for_buyer' in info:
                        divisor_client = divisor_client + 1.0
                    if 'feedback_for_provider' in info:
                        divisor_freelancer = divisor_freelancer + 1.0

                total_charge = 0
                total_hours = 0
                durations = []
                feedbacks = {}
                for info in assignment_info:
                    if 'total_charge' in info and self.is_float(info['total_charge']):
                        total_charge += float(info['total_charge'])
                    if 'tot_hours' in info and self.is_int(info['tot_hours']):
                        current_total_hours = int(info['tot_hours'])
                        total_hours += current_total_hours
                    else:
                        current_total_hours = None
                    if 'start_date' in info and 'end_data' in info \
                            and self.is_int(info['start_date']) and self.is_int(info['end_data']):
                        # duration is captured in weeks
                        duration = (datetime.fromtimestamp(int(info['end_data']) / 1000) -
                                    datetime.fromtimestamp(int(info['start_date']) / 1000)).days / 7.0
                        duration = int(round(duration, 0))
                        if 'workload' in job:
                            if job['workload'] == "Less than 10 hrs/week":
                                if current_total_hours:
                                    duration = self.clamp(duration, current_total_hours / 10, current_total_hours / 1)
                            elif job['workload'] == "10-30 hrs/week":
                                if current_total_hours:
                                    duration = self.clamp(duration, current_total_hours / 30, current_total_hours / 10)
                            elif job['workload'] == "30+ hrs/week":
                                if current_total_hours:
                                    duration = min(current_total_hours / 30, duration)
                        durations.append(duration)
                    elif current_total_hours and 'workload' in job:
                        if job['workload'] == "Less than 10 hrs/week":
                            duration = int(round(current_total_hours / 5.0, 0))
                        elif job['workload'] == "10-30 hrs/week":
                            duration = int(round(current_total_hours / 20.0, 0))
                        else:
                            duration = int(round(current_total_hours / 30.0, 0))
                        durations.append(duration)
                    if 'feedback_for_provider' in info:
                        for feedback in \
                                info['feedback_for_provider']['scores']['score']:
                            if 'feedback_for_freelancer_{}'.format(
                                    feedback['label'].lower()) in feedbacks:
                                feedbacks[
                                    'feedback_for_freelancer_{}'.format(feedback['label'].lower())] \
                                    += float(feedback['score']) / divisor_freelancer
                            else:
                                feedbacks[
                                    'feedback_for_freelancer_{}'.format(feedback['label'].lower())] \
                                    = float(feedback['score']) / divisor_freelancer
                    if 'feedback_for_buyer' in info:
                        for feedback in info['feedback_for_buyer']['scores']['score']:
                            if 'feedback_for_client_{}'.format(
                                    feedback['label'].lower()) in feedbacks:
                                feedbacks[
                                    'feedback_for_client_{}'.format(feedback['label'].lower())] \
                                    += float(feedback['score']) / divisor_client
                            else:
                                feedbacks[
                                    'feedback_for_client_{}'.format(feedback['label'].lower())] \
                                    = float(feedback['score']) / divisor_client
                if 'op_contractor_tier' in job_profile \
                        and self.is_int(job_profile['op_contractor_tier']):
                    job['experience_level'] = int(job_profile['op_contractor_tier'])
                job.update(feedbacks)
                job['freelancer_count'] = int(len(assignment_info))
                job['total_charge'] = float(total_charge)
                job['total_hours'] = total_hours
                job['duration_weeks_median'] = self.median(durations)
                job['duration_weeks_total'] = sum(durations)
            except Exception as e:
                print "Enriching job {} with additional data failed: {}" \
                    .format(job['id'], e)
        return found_jobs, job_profiles

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

        # connect to Upwork
        client = upwork.Client(public_key=credentials.public_key, secret_key=credentials.secret_key,
                               oauth_access_token=credentials.oauth_access_token,
                               oauth_access_token_secret=credentials.oauth_access_token_secret,
                               timeout=30)

        # assemble data in multiple iterations because of maximum number of data we can request
        for p in range(0, pages):

            if p == pages - 1:
                _sample_size = (sample_size % max_request_size) if (sample_size % max_request_size) != 0 else sample_size
            # print 'paging: ' + str(p * max_request_size) + ';' + str(_sample_size)

            query_data = {'q': '*', 'category2': 'Data Science & Analytics',
                          'job_status': 'completed', 'days_posted': days_posted}

            # try to get data until we either got it or we exceed the limit
            for i in range(0, max_tries):
                try:
                    if page_offset is None:
                        found_jobs.extend(
                            client.provider_v2.search_jobs(data=query_data, page_offset=(p * max_request_size),
                                                           page_size=_sample_size))
                    else:
                        found_jobs.extend(
                            client.provider_v2.search_jobs(data=query_data,
                                                           page_offset=page_offset + (p * max_request_size),
                                                           page_size=_sample_size))
                    print 'Successfully found jobs, page_offset=' + str(p * max_request_size) + ', page_size=' + str(
                        _sample_size)
                    exception = "None"
                    break
                except Exception as e:
                    print 'Number of tries for job search: ' + str(i)
                    print e
                    exception = str(e.code) + ' - ' + e.msg

        if found_jobs is not None:

            # Uncomment to load from files instead from Upwork

            # with open(working_dir + "job_profiles_complete.json") as f:
            #     job_profiles = json.load(f)
            # stored_profiles = {}
            # for job_profile in job_profiles:
            #     stored_profiles[job_profile['id']] = job_profile
            #
            # with open(working_dir + "found_jobs_4K.json") as f:
            #     found_jobs = json.load(f)
            #
            # found_jobs, job_profiles = self.enrich_with_job_profiles(client, found_jobs,
            #                                                          save_incrementally=False,
            #                                                          stored_job_profiles=stored_profiles)

            found_jobs, job_profiles = self.enrich_with_job_profiles(client, found_jobs,
                                                                     save_incrementally=True)

            with open(working_dir + strftime("found_jobs_%d.%m.-%H%M.json", gmtime()), "a+") as f:
                f.truncate()
                f.write(json.dumps(found_jobs))

            # store the jobs for all the mining modules as fallback
            with open(working_dir + "dm_lib/" + JOBS_FILE, "a+") as f:
                f.truncate()
                f.write(json.dumps(found_jobs))

            with open(working_dir + strftime("job_profiles_%d.%m.-%H%M.json", gmtime()), "a+") as f:
                f.truncate()
                f.write(json.dumps(job_profiles))

            response = rdb.table(RDB_JOB_TABLE).insert(found_jobs, conflict="replace").run(g.rdb_conn)

            if not rdb.table_list().contains(RDB_JOB_OPTIMIZED_TABLE).run(g.rdb_conn):
                rdb.table_create(RDB_JOB_OPTIMIZED_TABLE).run(g.rdb_conn)
            else:
                rdb.table(RDB_JOB_OPTIMIZED_TABLE).delete().run(g.rdb_conn)

            # prepare data with dm_lib
            data_frame = prepare_data(file_name='', jobs=found_jobs)
            data_frame.date_created = data_frame.date_created.apply(
                lambda time: time.to_pydatetime().replace(
                    tzinfo=rdb.make_timezone("+02:00"))
            )
            data_frame['id'] = data_frame.index
            rdb.table(RDB_JOB_OPTIMIZED_TABLE).insert(
                data_frame.to_dict('records'), conflict="replace").run(g.rdb_conn)

            # make all mining modules aware of the data refresh
            update_urls = [module_url + "update_model/" for module_url in self.mining_module_urls.values()]
            try:
                rs = (grequests.post(u) for u in update_urls)
                grequests.map(rs)
            except Exception as e:
                print "Exception: {}".format(e)

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
        last_updated = rdb.table(RDB_JOB_TABLE).order_by("requested_on").get_field("requested_on").max().run(g.rdb_conn)
        target_zone = tz.gettz('Europe/Zurich')
        last_updated = last_updated.replace(tzinfo=tz.gettz('UTC'))
        last_updated = last_updated.astimezone(target_zone)
        return "<h1>Data Module</h1><p>Last updated: {} </p>".format(last_updated)
    except Exception as e:
        return "<h1>Data Module</h1><p>Never updated</p>"


if __name__ == '__main__':
    db_setup()
    app.run(debug=True, use_debugger=False, use_reloader=False, host="0.0.0.0")
