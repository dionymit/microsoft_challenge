from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
app = Flask(__name__)
import mysql.connector
from mysql.connector import errorcode

config = {
  'host':'aiyoungsters.mysql.database.azure.com',
  'user':'aiyoungsters',
  'password':'AI12345#',
  'database':'community',
  'client_flags': [mysql.connector.ClientFlag.SSL],
  'ssl_ca': 'DigiCertGlobalRootG2.crt.pem'
}  
conn = mysql.connector.connect(**config)
print("Connection established")

my_cursor = conn.cursor()

@app.route('/db')
def db():
    my_cursor.execute('SELECT * FROM accounts;')
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