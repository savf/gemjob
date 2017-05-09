import json
import os
import subprocess
from time import sleep

import requests
from flask import Flask, request
from flask_restful import Api
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

import credentials

os.environ['DISPLAY'] = ':0'

app = Flask(__name__)
api = Api(app)

max_tries = 10
max_request_size = 99

wait_between_html_extractions = 10  # in seconds


def get_web_content(found_jobs):
    login_successful = upwork_login()
    if login_successful:
        print 'get_web_content: Login successful'
        bad_jobs = []
        for job in found_jobs:
            sleep(wait_between_html_extractions)  # wait first, to avoid DOSing Upwork

            url = job['url']
            # get_result = session.get(url, headers={
            #    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:53.0) Gecko/20100101 Firefox/53.0'})
            driver.get(url)

            elements = driver.find_elements_by_css_selector('p.m-xs-bottom')
            if len(elements) > 1:

                for element in elements:
                    split_text = element.text.split(':')
                    job[split_text[0]] = split_text[1]
            else:
                print 'get_web_content: Page contains no info'

                bad_jobs.append(job)

        # Only return jobs which have been augmented with data from the web
        found_jobs[:] = [job for job in found_jobs if job not in bad_jobs]
        return found_jobs

    print 'get_web_content: No session'
    return None


def upwork_login():
    sleep(wait_between_html_extractions)  # wait first, to avoid DOSing Upwork

    upwork_login_url = 'https://www.upwork.com/ab/account-security/login'
    # Don't log in again, if already logged in
    if 'upwork' in driver.current_url and \
            len(driver.find_elements_by_class_name('organization-selector')) > 0:
        return True

    driver.get(upwork_login_url)

    login_token = driver.find_element_by_id('login__token')

    if login_token is not None:
        login_token = login_token.get_attribute('value')
        driver.find_element_by_id('login_username').send_keys(credentials.login_username)
        driver.find_element_by_id('login_password').send_keys(credentials.login_password)
        driver.find_elements_by_class_name('checkbox-replacement-helper')[0].click()

        driver.find_element_by_name("login").submit()

        if len(driver.find_elements_by_class_name('organization-selector')) > 0:
            return True
    #else:
    #    with open(working_dir + "login_page.html", "a+") as f:
    #        f.truncate()
    #        f.write(soup.encode("utf-8"))
    #    self.session_requests = None
    print 'upwork_login: Login failed'
    return False

@app.route('/', methods=['POST'])
def index():
    json_payload = request.get_json(force=True)
    found_jobs = get_web_content(json_payload)

    return json.dumps(found_jobs)


if __name__ == '__main__':
    chrome_options = Options()
    chrome_options.add_experimental_option('prefs', {
        'credentials_enable_service': False,
        'profile': {
            'password_manager_enabled': False
        }
    })
    driver = webdriver.Chrome(chrome_options=chrome_options)
    app.run(debug=True, use_debugger=False, use_reloader=False, host="0.0.0.0")
