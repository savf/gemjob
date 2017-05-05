import os
from flask import Flask, render_template, request, make_response, redirect, session
import re
import requests
import upwork
import credentials

working_dir = os.path.dirname(os.path.realpath(__file__)) + '/'
os.environ['HTTPLIB_CA_CERTS_PATH'] = working_dir + 'cacert.pem'

app = Flask(__name__)

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

        first_name = user_info['auth_user']['first_name']
        last_name = user_info['auth_user']['last_name']
        return render_template("index.html", first_name=first_name, last_name=last_name)
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


if __name__ == '__main__':
    app.secret_key = 'xyz'
    app.run(debug=True, host="0.0.0.0")
