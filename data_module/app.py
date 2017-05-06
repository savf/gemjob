# Data Module
# Request sample to be stored as JSON using:
# -> http://localhost:5000/update_data/{sample size}
# e.g.
# -> http://localhost:5000/update_data/250
# if sample size is not provided, a default number will be chosen
# -> http://localhost:5000/update_data/

import json
import os

from datetime import datetime
from dateutil import tz
from time import sleep, strftime, gmtime

import requests
import rethinkdb as rdb
import upwork
from bs4 import BeautifulSoup
from flask import Flask, g, abort
from flask_restful import Resource, Api
from rethinkdb.errors import RqlRuntimeError, RqlDriverError

import credentials

working_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
os.environ['HTTPLIB_CA_CERTS_PATH'] = working_dir + 'cacert.pem'

app = Flask(__name__)
api = Api(app)

RDB_HOST = 'database_module'
RDB_PORT = 28015
RDB_DB = 'datasets'
RDB_TABLE = 'jobs'

max_tries = 10
max_request_size = 99

wait_between_html_extractions = 10  # in seconds


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

    ### get request
    def get(self, sample_size=5, days_posted=1):

        if sample_size < 1:
            return {'api_name': 'Data module REST API', 'success': False, 'sample-size': 0,
                    'exception': 'sample_size too small'}
        if days_posted < 1:
            return {'api_name': 'Data module REST API', 'success': False, 'sample-size': sample_size,
                    'exception': 'Only non-zero and positive values for days posted allowed'}

        found_jobs = []
        pages = 1 + (sample_size - 1) / max_request_size
        print 'pages: ' + str(pages)
        _sample_size = max_request_size

        exception = 'none'

        # assemble data in multiple iterations because of maximum number of data we can request
        for p in range(0, pages):

            if p == pages - 1:
                _sample_size = sample_size % max_request_size
            # print 'paging: ' + str(p * max_request_size) + ';' + str(_sample_size)

            # connect to Upwork
            client = upwork.Client(public_key=credentials.public_key, secret_key=credentials.secret_key,
                                   oauth_access_token=credentials.oauth_access_token,
                                   oauth_access_token_secret=credentials.oauth_access_token_secret,
                                   timeout=30)

            query_data = {'q': '*', 'category2': 'Data Science & Analytics',
                          'job_status': 'completed', 'days_posted': days_posted}

            # try to get data until we either got it or we exceed the limit
            for i in range(0, max_tries):
                try:
                    found_jobs.extend(
                        client.provider_v2.search_jobs(data=query_data, page_offset=(p * max_request_size),
                                                       page_size=_sample_size))
                    print 'Successfully found jobs, page_offset=' + str(p * max_request_size) + ', page_size=' + str(
                        _sample_size)
                    break
                except Exception as e:
                    print 'Num of tries: ' + str(i)
                    print e
                    exception = str(e.code) + ' - ' + e.msg

        # get additional info from webpages
        found_jobs = self.get_web_content(found_jobs)
        if found_jobs != None:
            # data to json
            found_jobs_json = json.dumps(found_jobs)

            # add current time as timestamp to all jobs
            for job in found_jobs:
                job.update({"requested_on": rdb.now()})

            response = rdb.table(RDB_TABLE).insert(found_jobs, conflict="replace").run(g.rdb_conn)

            with open(working_dir + strftime("found_jobs_%d.%m.-%H%M.json", gmtime()), "a+") as f:
                f.truncate()
                f.write(found_jobs_json)
            success_criterion = len(found_jobs) == _sample_size and response['inserted'] > 0
            return {'api_name': 'Data module REST API', 'success': success_criterion,
                    'sample-size': len(found_jobs), 'exception': exception, 'database-response': response}

        return {'api_name': 'Data module REST API', 'success': False,
                'sample-size': 0, 'exception': 'Web crawling failed'}

    ### get end

    ### get info from HTML pages
    def get_web_content(self, found_jobs):
        session = self.get_upwork_page_session()
        if session:
            print 'get_web_content: Login successful'
            bad_jobs = []
            for job_data in found_jobs:
                sleep(wait_between_html_extractions)  # wait first, to avoid DOSing Upwork

                url = job_data['url']
                #get_result = session.get(url, headers={
                #    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:53.0) Gecko/20100101 Firefox/53.0'})
                get_result = session.get(url, headers={
                    'User-Agent': credentials.headers['User-Agent']})
                soup = BeautifulSoup(get_result.text)

                elements = soup.find_all('p', {'class': 'm-xs-bottom'})
                if len(elements) > 1:
                    for element in elements:
                        split_text = element.get_text(strip=True).split(':')
                        job_data[split_text[0]] = split_text[1]
                else:
                    if any("captcha" in (item['src'] if hasattr(item, 'src') else "")
                           for item in soup.find_all('script')):
                        print 'get_web_content: Page blocked by captcha'
                    else:
                        print 'get_web_content: Page contains no info'

                    bad_jobs.append(job_data)

            # Only return jobs which have been augmented with data from the web
            found_jobs[:] = [job for job in found_jobs if job not in bad_jobs]
            return found_jobs

        print 'get_web_content: No session'
        return None

    ### get_web_content end

    ### login and get session
    def get_upwork_page_session(self):
        sleep(wait_between_html_extractions)  # wait first, to avoid DDOSing Upwork

        self.session_requests = requests.session()

        upwork_login_url = 'https://www.upwork.com/ab/account-security/login'
        login_page = self.session_requests.get(upwork_login_url, headers={
                    'User-Agent': credentials.headers['User-Agent']})

        soup = BeautifulSoup(login_page.text)
        login_token = soup.find(id="login__token")

        if login_token != None:
            login_token = login_token['value']
            payload = {'login[username]': credentials.login_username,
                       'login[password]': credentials.login_password,
                       'login[rememberme]': 1,
                       'login[_token]': login_token,
                       'login[iovation]': ''
                       }
            login_response = self.session_requests.post(
                upwork_login_url,
                data=payload,
                headers={'User-Agent': credentials.headers['User-Agent']}
            )
            if login_response.ok:
                return self.session_requests
        else:
            with open(working_dir + "login_page.html", "a+") as f:
                f.truncate()
                f.write(soup.encode("utf-8"))
            self.session_requests = None
        print 'get_upwork_page_session: Login failed'
        return None
        ### get_upwork_page_session end


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


api.add_resource(DataUpdater, '/update_data/', '/update_data/<int:sample_size>/<int:days_posted>')


@app.route('/')
def start():
    last_updated = rdb.table(RDB_TABLE).order_by("requested_on").get_field("requested_on").max().run(g.rdb_conn)
    target_zone = tz.gettz('Europe/Zurich')
    last_updated = last_updated.replace(tzinfo=tz.gettz('UTC'))
    last_updated = last_updated.astimezone(target_zone)
    return "<h1>Data Module</h1><p>Last updated: {} </p>".format(last_updated)


if __name__ == '__main__':
    db_setup()
    app.run(debug=True, use_debugger=False, use_reloader=False, host="0.0.0.0")
