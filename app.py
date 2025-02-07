from flask import Flask, request, render_template
import sys
import os
import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application=Flask(__name__)
app=application

#route for home page
@app.route('/')
def index():

    # Flask to render and return the index.html template when someone visits the root URL. 
    # So, when users go to your app's home page (/), they'll see the contents of index.html.
    return render_template('index.html')

@app.route("/predictdata",methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race/ethnicity'),
            parental_level_of_education=request.form.get('parental level of education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test preparation course'),
            writing_score=float(request.form.get('writing score')),
            reading_score=float(request.form.get('reading score')),

        )



        pred_df=data.get_data_as_dataframe()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=round(results[0],2))
    

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True) 


#to run this project just type python app.py in terminal
# open explorer type 127.0.0.1:5000, this opens index.html page
# then type 127.0.0.1:5000/predictdata, this opens home.html