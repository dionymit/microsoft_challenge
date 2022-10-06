from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
app = Flask(__name__)
import psycopg2


conn = psycopg2.connect(database = "community", user = "aiyoungsters", password = "AI12345#", 
host = "aiyoungsters.postgres.database.azure.com", port = "5432", sslmode='require')
print("Opened database successfully")

my_cursor = conn.cursor()

#CREATE DATABASE tracks (id int NOT NULL AUTO_INCREMENT, routeid int, latitude float, longtitude float, timestamp int, PRIMARY KEY(id))
#CREATE DATABASE routs (id int NOT NULL AUTO_INCREMENT, route_name VARCHAR(1000), account_id int, track_id int, favorite bool, PRIMARY KEY(id))
#CREATE DATABASE accounts (id int NOT NULL AUTO_INCREMENT, name VARCHAR(1000), mail VARCHAR(1000), PRIMARY KEY(id))

@app.route('/filldb/<a1>/<a2>')
def filldb(a1,a2):
    f = open(f"out_r{a1}_v{a2}.txt", "r")
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
    my_cursor.execute(f'SELECT latitude,longtitude FROM tracks WHERE routeid={id};')
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
    return render_template('route.html')


if __name__ == '__main__':
   app.run(debug=True)