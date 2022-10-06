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
        my_cursor.execute(x+ ";")

    my_cursor.execute('SELECT * FROM tracks;')
    myresult = my_cursor.fetchall()
    return myresult


@app.route('/db')
def db():
    my_cursor.execute('SELECT * FROM accounts;')
    myresult = my_cursor.fetchall()
    return myresult
@app.route('/db')
def db():
    my_cursor.execute('SELECT * FROM tracks;')
    myresult = my_cursor.fetchall()
    return myresult

@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/hello', methods=['POST'])
def hello():
   name = request.form.get('name')

   if name:
       print('Request for hello page received with name=%s' % name)
       return render_template('hello.html', name = name)
   else:
       print('Request for hello page received with no name or blank name -- redirecting')
       return redirect(url_for('index'))


if __name__ == '__main__':
   app.run()