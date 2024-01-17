from flask import Flask, render_template, redirect, request
import pickle
import numpy as np
import os


# Importing the model

with open(os.path.join(r"model", 'dl_ANN.pkl'), 'rb') as file:
    model = pickle.load(file)

# starting the application
    
app = Flask(__name__)

# Creating the default route
@app.route('/')
def index():
    return render_template('home.html')

# Getting the data witch you entered in web
@app.route('/predict', methods=['POST'])
def home():
    data1 = float(request.form['credit'])
    data2 = float(request.form['age'])
    data3 = float(request.form['tenure'])
    data4 = float(request.form['balance'])
    data5 = float(request.form['products'])
    data6 = float(request.form['cc'])
    data7 = float(request.form['active'])
    data8 = float(request.form['salary'])
    data9 = float(request.form['germany'])
    data10 = float(request.form['spain'])
    data11 = float(request.form['gender'])
    arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11]])
    pred = model.predict(arr)
    return render_template('generate.html', data=pred)

if __name__ == "__main__":
    app.run(debug=True)



