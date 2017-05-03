from flask import Flask, render_template, request
import re

app = Flask(__name__)

@app.route('/')
def start():
    return render_template("login.html")

@app.route('/', methods=['POST'])
def login():
    upwork_email = request.form['upwork_email']
    upwork_password = request.form['upwork_password']
    if len(upwork_email) == 0 or len(upwork_password) == 0:
        return render_template("login.html", server_message="Please provide email and password",
            upwork_email=upwork_email, upwork_password=upwork_password)
    elif not re.match(r"[^@]+@[^@]+\.[^@]+", upwork_email):
        return render_template("login.html", 
            server_message="Please provide a valid email address",
            upwork_email=upwork_email, upwork_password=upwork_password)
    return "You signed in as: <br>" + upwork_email + "<br>" + upwork_password + "<br><br><br><br> ... just kidding, this did nothing"

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
