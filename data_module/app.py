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
import json
import credentials

app = Flask(__name__)
api = Api(app)

max_tries = 10
max_request_size = 99

class DataUpdater(Resource):  # Our class "DataUpdater" inherits from "Resource"
    # get request
    def get(self, sample_size=50):

        if sample_size < 1:
            return {'api_name': 'Data module REST API', 'success': False, 'sample-size': 0, 'exception': 'sample_size too small'}

        found_jobs = []
        pages = 1 + (sample_size-1) / max_request_size
        print 'pages: ' + str(pages)
        _sample_size = max_request_size

        exception = 'none'

        # assemle data in multiple iterations because of maximum number of data we can request
        for p in range(0, pages):

            if p == pages-1:
                _sample_size = sample_size % max_request_size

            print 'paging: ' + str(p * max_request_size) + ';' + str(_sample_size)

            # connect to upwork with Benjamin's account and API key

            client = upwork.Client(public_key=credentials.public_key, secret_key=credentials.secret_key,
                                   oauth_access_token=credentials.oauth_access_token,
                                   oauth_access_token_secret=credentials.oauth_access_token_secret,
                                   timeout=30)

            query_data = {'q': '*', 'category2': 'Data Science & Analytics', 'job_status': 'completed'}

            # try to get data until we either got it or we exceed the limit
            for i in range(0, max_tries):
                try:
                    found_jobs.extend(client.provider_v2.search_jobs(data=query_data, page_offset=(p * max_request_size), page_size=_sample_size))
                    break
                except Exception as e:
                    print e
                    exception = str(e.code) + ' - ' + e.msg

        # data to json
        found_jobs_json = json.dumps(found_jobs)

        # TODO store found_jobs in DB
        with open(working_dir+"found_jobs.json", "a+") as f:
            f.truncate()
            f.write(found_jobs_json)

        return {'api_name': 'Data module REST API', 'success': len(found_jobs) == _sample_size, 'sample-size': len(found_jobs), 'exception': exception}


api.add_resource(DataUpdater, '/update_data/', '/update_data/<int:sample_size>')

@app.route('/')
def start() :
    last_updated = 'get this from db'
    return "<h1>Data Module</h1><p>Last updated: "+ last_updated +"</p>"

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
