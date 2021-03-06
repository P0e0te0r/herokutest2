from flask import Flask, redirect, url_for, render_template, request, redirect, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, DecimalField, SelectField, RadioField, FloatField, IntegerField
from wtforms.validators import DataRequired
import smtplib
import os
from datetime import timedelta, datetime
import sqlite3
import pandas as pd
#from Web_app_2 import ml_module

app = Flask(__name__)

#Create Secret Key
app.config['SECRET_KEY']="retggrre1"

#app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///Iris_Class.db"
conn = sqlite3.connect('Iris_Class.db', check_same_thread=False)
c = conn.cursor()
#c.execute('CREATE TABLE iris_class_db'\
#          '(id INTEGER PRIMARY KEY, Petal_length REAL, Petal_width REAL, Sepal_length REAL, Sepal_width REAL, iris_class STR, Pred_Acc REAL)')
conn.commit()
conn.close()   
# Initialize the app
db = SQLAlchemy(app)


class FlowerForm(FlaskForm):
    Petal_length = IntegerField("Petal length", validators =[DataRequired()])
    Petal_width = IntegerField("Petal length", validators =[DataRequired()])
    Sepal_length = IntegerField("Petal length", validators =[DataRequired()])
    Sepal_width = IntegerField("Petal length", validators =[DataRequired()])
    submit = SubmitField("Submit")
        

class FeedbackForm(FlaskForm):
    feedback = RadioField(label = "Feedback", choices = ["Prediction ist RICHTIG", "Prediction ist FALSCH"], validators =[DataRequired()])
    submit = SubmitField("Submit")



@app.route("/")
def welcome():
    title = "Willkommen auf der Blüten-Bestimmungs-Website"
    return render_template("welcome.html")

@app.route("/home", methods = ["POST", "GET"])
def home():
    title = "Eingabe Template"
    Petal_length = None
    Petal_width = None
    Sepal_length = None
    Sepal_width = None
    form = FlowerForm()
    proba = None
    clas = None
    if form.validate_on_submit():
        Petal_length = form.Petal_length.data
        Petal_width = form.Petal_width.data
        Sepal_length = form.Sepal_length.data
        Sepal_width = form.Sepal_width.data
        
        form.Petal_length.data = ''
        form.Petal_width.data = ''
        form.Sepal_length.data = ''
        form.Sepal_width.data = ''
        session["proba"] = None
        session["iris_class"] = None
        
        
        import pandas as pd
        import numpy as np
        import os
        import matplotlib.pyplot as plt
        import seaborn as sns
        import warnings
        
        warnings.filterwarnings('ignore')
                      
        array = ([[Petal_length, Petal_width, Sepal_length, Sepal_width]])
        import pickle
        with open(r"C:\Users\Test\Documents\Machine_Learning\Web_Anwendung\model_pickle",'rb') as file:
            mp = pickle.load(file)
            proba = mp.predict_proba(array).max()
            clas = mp.predict(array)
        session["proba"] = proba
       
               
        label = {0: "Iris setosa", 1: "Iris virginica", 2: "Iris gloriosa" }
        iris_class = label[clas[0]]       
        conn = sqlite3.connect('Iris_Class.db')
        c = conn.cursor()
        c.execute("INSERT INTO iris_class_db (Petal_length, Petal_width, Sepal_length, Sepal_width, iris_class) VALUES (?,?,?,?,?)",(Petal_length, Petal_width, Sepal_length, Sepal_width, iris_class))
        letzte_zeile = c.lastrowid
        conn.commit()
        conn.close()
        session["iris_class"] = iris_class
        return render_template("result_feedback.html", 
                               proba = proba, 
                               iris_class = iris_class,
                               form = form, 
                               Petal_length = Petal_length, 
                               Sepal_width = Sepal_width,
                               Sepal_length = Sepal_length, 
                               Petal_width = Petal_width)
    return render_template("home.html", form = form, Petal_width = Petal_width, title=title)

        
                           
@app.route("/result_feedback", methods = ["POST", "GET"])
def result():
    Petal_length = request.form.get("Petal_length")
    Petal_width  = request.form.get("Petal_width ")
    Sepal_length = request.form.get("Sepal_length")
    Sepal_width = request.form.get("Sepal_width")
    
    

    

@app.route('/test', methods = ["POST", "GET"])
def feedback():
    form_fb = FeedbackForm()
    if form_fb.validate_on_submit():
        feedback = form_fb.feedback.data
        
        if feedback == "Prediction ist FALSCH":
            Pred_Acc = 0
            conn = sqlite3.connect('Iris_Class.db')
            c = conn.cursor()
            
            c.execute("UPDATE iris_class_db SET Pred_Acc=0 WHERE id = (SELECT MAX(id) FROM iris_class_db)")
            conn.commit()
            conn.close()    
        else:
            Pred_Acc = 1
            conn = sqlite3.connect('Iris_Class.db')
            c = conn.cursor()            
            #letzte_zeile = c.lastrowid
            c.execute("UPDATE iris_class_db SET Pred_Acc=1 WHERE id = (SELECT MAX(id) FROM iris_class_db)")
            conn.commit()
            conn.close()    
        #conn = sqlite3.connect('Classification.db')
        #c = conn.cursor()
        #c.execute("INSERT INTO classification_db (Pred_Acc) VALUES (Pred_Acc)")
        #conn.commit()
        #conn.close()    
        return render_template("exit.html", Pred_Acc = Pred_Acc)
    return render_template("test.html", form_fb = form_fb)
                               



if __name__ == "__main__":
    app.run(debug=True)
    


    