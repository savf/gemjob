import os
working_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
os.environ['HTTPLIB_CA_CERTS_PATH'] = working_dir + 'cacert.pem'
from flask import Flask, render_template, request, make_response, redirect, session, jsonify
import requests
import upwork
import credentials
import datetime
from pretty_print import *
# module_urls = {'D': 'http://data_module:5000/', 'DM': 'http://data_mining_module:5000/', 'DB': 'http://database_module:8080/', 'CL': 'http://cluster_module:5002/'}
module_urls = {'D': 'http://localhost:5000/', 'DM': 'http://localhost:5001/', 'DB': 'http://localhost:8001/', 'CL': 'http://localhost:5002/'}

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
def job():
    try:
        access_token = session['access_token']
        access_token_secret = session['access_token_secret']
        client = upwork.Client(public_key=credentials.public_key, secret_key=credentials.secret_key,
                               oauth_access_token=access_token,
                               oauth_access_token_secret=access_token_secret,
                               timeout=30)
        user_info = session.get('user_info')
        profile_pic = user_info['info']['portrait_32_img']
        skills_list = client.provider.get_skills_metadata()
        print "### Skills list:"
        # pretty_print(skills_list)
        skills_list_js = '['
        for skill in skills_list:
            skills_list_js = skills_list_js + '"' + str(skill) + '",'
        skills_list_js = skills_list_js[0:-1] + ']'
        print skills_list_js
        return render_template("job.html", profile_pic=profile_pic, current_date=datetime.date.today().strftime("%m-%d-%Y"), skills_list=skills_list_js)
    except Exception as err:
        print err
        session.clear()
        return render_template("login.html")

# @app.route('/job/id=<string:id>')
# def job_existing(id):
#     try:
#         user_info = session.get('user_info')
#         profile_pic = user_info['info']['portrait_32_img']
#         return render_template("job.html", profile_pic=profile_pic, current_date=datetime.date.today().strftime("%m-%d-%Y"), job_id=id)
#     except Exception as err:
#         print err
#         session.clear()
#         return render_template("login.html")

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

if __name__ == '__main__':
    app.secret_key = 'xyz'
    app.run(debug=True, use_debugger=False, use_reloader=False, host="0.0.0.0", port=8000)
