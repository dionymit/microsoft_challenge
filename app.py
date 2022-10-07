from datetime import datetime
import pkgutil
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
app = Flask(__name__)
import psycopg2
import pickle as pkl
import numpy as np
import pandas as pd

conn = psycopg2.connect(database = "community", user = "aiyoungsters", password = "AI12345#", 
host = "aiyoungsters.postgres.database.azure.com", port = "5432", sslmode='require')
print("Opened database successfully")

my_cursor = conn.cursor()

#3 Tables in hierachical order
#CREATE TABLE tracks (id int NOT NULL AUTO_INCREMENT, routeid int, latitude float, longtitude float, timestamp int, PRIMARY KEY(id))
#CREATE TABLE routs (id int NOT NULL AUTO_INCREMENT, route_name VARCHAR(1000), account_id int, track_id int, favorite bool, PRIMARY KEY(id))
#CREATE TABLE accounts (id int NOT NULL AUTO_INCREMENT, name VARCHAR(1000), mail VARCHAR(1000), PRIMARY KEY(id))


#populate the DB with track data
@app.route('/filldb/<a1>/<a2>')
def filldb(a1,a2):
    f = open(f"./track_data/out_r{a1}_v{a2}.txt", "r")
    for x in f:
        print(x + ";")
        my_cursor.execute(x)
        conn.commit()

    my_cursor.execute('SELECT * FROM tracks;')
    myresult = my_cursor.fetchall()
    return myresult



@app.route('/routes_all')
def routes_all():
    my_cursor.execute('SELECT COUNT(*), route_name,MAX(track_id),MAX(distance) ,AVG(duration) FROM routes GROUP BY route_name;')
    myresult = my_cursor.fetchall()
    return myresult

@app.route('/tracks/<id>')
def tracks(id):
    my_cursor.execute(f'SELECT longtitude,latitude FROM tracks WHERE routeid={id};')
    myresult = my_cursor.fetchall()
    return myresult


@app.route('/db')
def db():
    my_cursor.execute('SELECT * FROM tracks JOIN routes ON tracks.routeid=routes.track_id;')
    myresult = my_cursor.fetchall()
    return myresult


@app.route('/')
def home():
   print('Request for home page received')
   return render_template('home.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/routes', methods=['GET','POST'])
def routes():
    print('Request for routes page received')
    if request.method=="POST":
        return redirect(url_for('partner'))
    return render_template('route.html')

curr_dist = 0
curr_duration = 0
@app.route('/partner/<distance>/<duration>')
def partner(distance,duration):
    global curr_dist
    global curr_duration
    curr_dist=distance
    curr_duration=duration
    return render_template('partner.html')

@app.route('/select_parameters')
def select_parameters():
    return render_template('select_parameters.html')
    
@app.route('/start_run')
def start_run():
    return render_template('run.html')

@app.route('/partner_prediction')
def partner_prediction():
    with open('./ai/model.pkl', 'rb') as f:
        model = pkl.load(f)

    df = pd.DataFrame(columns=['distance', 'duration'])
    print(curr_dist)
    print(curr_duration)
    df = df.append({'distance':curr_dist, 'duration':curr_duration}, ignore_index=True)
    predict = model.predict(df)[0]
    if predict ==0:
        predict = 2
    print(predict)
    my_cursor.execute(f'SELECT accounts.*, AVG(distance),AVG(duration) FROM accounts LEFT JOIN routes ON routes.account_id=accounts.id WHERE accounts.id={predict} GROUP BY accounts.id; ')
    print(f'predict {predict}' )
    myresult = my_cursor.fetchall()
    return myresult


if __name__ == '__main__':
    
   app.run(debug=False)