# Data Module
# Request sample to be stored as JSON using:
# -> http://localhost:5000/update_data/{sample size}
# e.g.
# -> http://localhost:5000/update_data/250
# if sample size is not provided, a default number will be chosen
# -> http://localhost:5000/update_data/

from flask import Flask, g
from flask_restful import Resource, Api
import os
working_dir = os.path.dirname(os.path.realpath(__file__))+'/'
os.environ['HTTPLIB_CA_CERTS_PATH'] = working_dir + 'cacert.pem'
import upwork
import requests
from bs4 import BeautifulSoup
import json
import time
import credentials

app = Flask(__name__)
api = Api(app)

max_tries = 10
max_request_size = 99

wait_between_html_extractions = 10 # in seconds

###### class begin
class DataUpdater(Resource):  # Our class "DataUpdater" inherits from "Resource"

    ### get request
    def get(self, sample_size=5):

        if sample_size < 1:
            return {'api_name': 'Data module REST API', 'success': False, 'sample-size': 0, 'exception': 'sample_size too small'}

        found_jobs = []
        pages = 1 + (sample_size-1) / max_request_size
        print 'pages: ' + str(pages)
        _sample_size = max_request_size

        exception = 'none'

        # assemble data in multiple iterations because of maximum number of data we can request
        for p in range(0, pages):

            if p == pages-1:
                _sample_size = sample_size % max_request_size
            # print 'paging: ' + str(p * max_request_size) + ';' + str(_sample_size)

            # connect to Upwork
            client = upwork.Client(public_key=credentials.public_key, secret_key=credentials.secret_key,
                                   oauth_access_token=credentials.oauth_access_token,
                                   oauth_access_token_secret=credentials.oauth_access_token_secret,
                                   timeout=30)

            query_data = {'q': '*', 'category2': 'Data Science & Analytics', 'job_status': 'completed'}

            # try to get data until we either got it or we exceed the limit
            for i in range(0, max_tries):
                try:
                    found_jobs.extend(client.provider_v2.search_jobs(data=query_data, page_offset=(p * max_request_size), page_size=_sample_size))
                    print 'Successfully found jobs, page_offset=' + str(p * max_request_size) + ', page_size=' + str(_sample_size)
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

            # TODO store found_jobs in DB
            with open(working_dir+"found_jobs.json", "a+") as f:
                f.truncate()
                f.write(found_jobs_json)

            return {'api_name': 'Data module REST API', 'success': len(found_jobs) == _sample_size, 'sample-size': len(found_jobs), 'exception': exception}

        return {'api_name': 'Data module REST API', 'success': False,
                'sample-size': 0, 'exception': 'Web crawling failed'}
    ### get end

    ### get info from HTML pages
    def get_web_content(self, found_jobs):
        session = self.get_upwork_page_session()
        if session:
            print 'get_web_content: Login successful'
            for job_data in found_jobs:
                time.sleep(wait_between_html_extractions) # wait first, to avoid DDOSing Upwork

                url = job_data['url']
                get_result = session.get(url, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:53.0) Gecko/20100101 Firefox/53.0'})
                soup = BeautifulSoup(get_result.text, "lxml")

                elements = soup.find_all('p', {'class': 'm-xs-bottom'})
                if len(elements) > 1:
                    for element in elements:
                        split_text = element.get_text(strip=True).split(':')
                        job_data[split_text[0]] = split_text[1]
                else:
                    print 'get_web_content: Page contains no info'
                    return None

            return found_jobs

        print 'get_web_content: No session'
        return None
    ### get_web_content end

    ### login and get session
    def get_upwork_page_session(self):
        time.sleep(wait_between_html_extractions)  # wait first, to avoid DDOSing Upwork

        self.session_requests = requests.session()

        upwork_login_url = 'https://www.upwork.com/ab/account-security/login'
        login_page = self.session_requests.get(upwork_login_url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:53.0) Gecko/20100101 Firefox/53.0'})

        soup = BeautifulSoup(login_page.text, "lxml")
        login_token = soup.find(id="login__token")

        if login_token != None:
            login_token = login_token['value']
            payload = {     'login[username]': credentials.login_username,
                            'login[password]': credentials.login_password,
                            'login[rememberme]': 1,
                            'login[_token]': login_token,
                            'login[iovation]': ''
                            }
            login_response = self.session_requests.post(
                upwork_login_url,
                data = payload,
                headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:53.0) Gecko/20100101 Firefox/53.0'}
            )
            if login_response.ok:
                return self.session_requests
        else:
            with open(working_dir+"login_page.html", "a+") as f:
                f.truncate()
                f.write(soup.encode("utf-8"))
            self.session_requests = None
        print 'get_upwork_page_session: Login failed'
        return None
    ### get_upwork_page_session end

###### class end


api.add_resource(DataUpdater, '/update_data/', '/update_data/<int:sample_size>')

@app.route('/')
def start() :
    last_updated = 'get this from db'
    return "<h1>Data Module</h1><p>Last updated: "+ last_updated +"</p>"

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
