import os
working_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
os.environ['HTTPLIB_CA_CERTS_PATH'] = working_dir + 'cacert.pem'
from flask import Flask, render_template, request, make_response, redirect, session, jsonify
import requests
import upwork
import credentials
import datetime
from pretty_print import *
# module_urls = {'D': 'http://data_module:5000/', 'DM': 'http://data_mining_module:5000/', 'DB': 'http://database_module:8080/', 'CL': 'http://cluster_module:5002/', 'BU': 'http://budget_module:5003/', 'FE': 'http://feedback_module:5004/', 'JO': 'http://jobtype_module:5005/'}
module_urls = {'D': 'http://localhost:5000/', 'DM': 'http://localhost:5001/', 'DB': 'http://localhost:8001/', 'CL': 'http://localhost:5002/', 'BU': 'http://localhost:5003/', 'FE': 'http://localhost:5004/', 'JO': 'http://localhost:5005/'}

GLOBAL_VARIABLE = {}

app = Flask(__name__)


def get_jobs(client, teams):
    jobs = []
    try:
        for team in teams:
            ref = team['reference']
            team_jobs = client.hr.get_jobs(ref, include_sub_teams=1)
            if 'job' in team_jobs:
                if isinstance(team_jobs['job'], list):
                    for j in team_jobs['job']:
                        jobs.append(j)
                elif isinstance(team_jobs['job'], dict):
                    jobs.append(team_jobs['job'])
    except Exception as err:
        print err
    return jobs


def get_client_data(client):
    client_info = {}
    try:
        me = client.hr.get_user_me()
        user_info = session['user_info']

        details = client.provider.get_provider(me["profile_key"])
        pretty_print(details)
        client_info["client_country"] = details.get("dev_country", 0)
        # default values if no jobs are available, they hold
        client_info["client_jobs_posted"] = 0
        client_info["client_past_hires"] = 0
        if 'teams' in user_info:
            jobs_list = get_jobs(client, user_info['teams'])
            client_info["client_jobs_posted"] = len(jobs_list)
            if len(jobs_list) > 0:
                for job in jobs_list:
                    try:
                        if 'reference' in job:
                            job_profile = client.job.get_job_profile(str(
                                job['reference']))
                        elif 'job_ref_ciphertext' in job:
                            job_profile = client.job.get_job_profile(str(
                                job['job_ref_ciphertext']))
                        if job_profile is not None:
                            if 'buyer' in job_profile and 'op_tot_asgs' in job_profile['buyer']:
                                client_info["client_past_hires"] = job_profile['buyer']['op_tot_asgs']
                                break
                    except:
                        continue

        client_info["client_reviews_count"] = details.get("dev_tot_feedback", 0)
        if details.has_key("feedback"):
            client_info["client_feedback"] = details.get("feedback").get("score", 0)
        else:
            client_info["client_feedback"] = 0
    except Exception as err:
        print err
    return client_info


@app.route('/')
def start():
    try:
        access_token = session['access_token']
        access_token_secret = session['access_token_secret']
        client = upwork.Client(public_key=credentials.public_key, secret_key=credentials.secret_key,
                               oauth_access_token=access_token,
                               oauth_access_token_secret=access_token_secret,
                               timeout=30)

        user_info = session.get('user_info')
        if not user_info:
            user_info = client.auth.get_info()
            session['user_info'] = user_info
        user_info['teams'] = client.hr.get_teams() # get this info everytime, might have changed
        # pretty_print(user_info)
        first_name = user_info['auth_user']['first_name']
        last_name = user_info['auth_user']['last_name']
        profile_pic = user_info['info']['portrait_32_img']

        jobs_list = get_jobs(client, user_info['teams'])
        # print "### Jobs:"
        # pretty_print(jobs_list)
        return render_template("index.html", first_name=first_name, last_name=last_name, profile_pic=profile_pic, jobs_list=jobs_list)
    except Exception as err:
        print err
        session.clear()
        return render_template("login.html")


@app.route('/login')
def login():
    client = upwork.Client(public_key=credentials.public_key, secret_key=credentials.secret_key, timeout=30)
    request_token, request_token_secret = client.auth.get_request_token()
    session['request_token'] = request_token
    session['request_token_secret'] = request_token_secret
    authorize_url = client.auth.get_authorize_url(callback_url=request.base_url+"/upwork")
    return redirect(authorize_url)


@app.route('/login/upwork', methods=['GET'])
def login_callback():
    oauth_token = request.args.get('oauth_token', None)
    oauth_verifier = request.args.get('oauth_verifier', None)
    client = upwork.Client(public_key=credentials.public_key, secret_key=credentials.secret_key, timeout=30)
    client.auth.request_token = session['request_token']
    client.auth.request_token_secret = session['request_token_secret']
    session.clear()
    access_token, access_token_secret = client.auth.get_access_token(oauth_verifier)
    session['access_token'] = access_token
    session['access_token_secret'] = access_token_secret
    return redirect("/")


@app.route('/logout')
def logout():
    session.clear()
    return redirect("/")


@app.route('/admin')
def admin():
    return render_template("admin.html")


@app.route('/job')
def job(client=None, update_job_info=None, warning=None):
    try:
        if not client:
            access_token = session['access_token']
            access_token_secret = session['access_token_secret']
            client = upwork.Client(public_key=credentials.public_key, secret_key=credentials.secret_key,
                                   oauth_access_token=access_token,
                                   oauth_access_token_secret=access_token_secret,
                                   timeout=30)
        user_info = session.get('user_info')
        profile_pic = user_info['info']['portrait_32_img']

        if not GLOBAL_VARIABLE.has_key("skills_list"):
            skills_list = client.provider.get_skills_metadata()
            print "### Skills list:"
            # pretty_print(skills_list)
            skills_list_js = '['
            for skill in skills_list:
                skills_list_js = skills_list_js + '"' + str(skill) + '",'
            skills_list_js = skills_list_js[0:-1] + ']'
            print skills_list_js
            GLOBAL_VARIABLE["skills_list"] = skills_list_js

        client_info = get_client_data(client)

        return render_template("job.html", profile_pic=profile_pic, current_date=datetime.date.today().strftime("%m-%d-%Y"), skills_list=GLOBAL_VARIABLE["skills_list"], client_info=client_info, update_job_info=update_job_info, warning=warning)
    except Exception as err:
        print err
        session.clear()
        return render_template("login.html")

@app.route('/job/id=<string:id>')
def job_existing(id):
    try:
        access_token = session['access_token']
        access_token_secret = session['access_token_secret']
        client = upwork.Client(public_key=credentials.public_key, secret_key=credentials.secret_key,
                               oauth_access_token=access_token,
                               oauth_access_token_secret=access_token_secret,
                               timeout=30)
        update_job_info = client.hr.get_job(id)
        pretty_print_dict(update_job_info)
        if update_job_info["category2"] != "Data Science & Analytics":
            return job(client=client, warning="The selected job is not a Data Science job!")
        return job(client=client, update_job_info=update_job_info)
    except Exception as err:
        print err
        session.clear()
        return render_template("login.html")


@app.route('/get_sample')
def get_sample():
    json_data = request.args.to_dict()
    for key, value in json_data.items():
        try:
            json_data[key] = int(value)
        except:
            json_data.pop(key)

    try:
        result = requests.post(module_urls['D']+"update_data/", json=json_data)
        return jsonify(result=result.content)
    except:
        return jsonify(result='Server not responding')


@app.route('/is_online')
def is_online():
    status = {'D': 'offline', 'DM': 'offline', 'DB': 'offline', 'WS': 'online'}
    try:
        requests.get(module_urls['D'])
        status['D'] = 'online'
    except:
        pass
    try:
        requests.get(module_urls['DM'])
        status['DM'] = 'online'
    except:
        pass
    try:
        requests.get(module_urls['DB'])
        status['DB'] = 'online'
    except:
        pass
    return jsonify(result=status)


@app.route('/get_realtime_predictions')
def get_realtime_predictions():
    json_data = request.args.to_dict()
    try:
        result = requests.post(module_urls['CL']+"get_predictions/", json=json_data)
        return jsonify(result=result.content)
        # return jsonify(result="Not implemented")
    except:
        return None # jsonify(result='Server not responding')


@app.route('/get_model_predictions')
def get_model_predictions():
    json_data = request.args.to_dict()
    predictions = dict()
    for module in [module_urls['BU'], module_urls['FE'], module_urls['JO']]:
        try:
            result = requests.post(module + "get_predictions/", json=json_data)
            for key, value in result.json().iteritems():
                predictions[key] = value
        except Exception as e:
            print "Exception: {}".format(e)
    return jsonify(result=predictions)

if __name__ == '__main__':
    app.secret_key = 'xyz'
    app.run(debug=True, use_debugger=False, use_reloader=False, host="0.0.0.0", port=8000)
