from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import json


app = Flask(__name__)

pickle_in=open('classifier.pkl','rb')
classifier = pickle.load(pickle_in)

# scoring
def score_record(data, classifier):

    return classifier.predict(data)[0], classifier.predict_proba(data)[:,1][0]

@app.route('/')
def hello_world():
    return 'Hello world! How are you?'
    
@app.route('/predict', methods=['GET','POST'])
def predict_score():
    
    #Recuperation des données 
    data = request.get_json()
    data = np.array(data)
    data = data.reshape(1, -1)

    # utilisation des données pour faire une prediction
    prediction, probability = score_record(data, classifier)
    data_df = pd.DataFrame(columns=['prediction','probability'])
    data_df = data_df.append({'prediction' : prediction}, ignore_index=True)
    data_df['probability'] = probability
    output = data_df.to_dict(orient='rows')
    
    return  jsonify(output)


if __name__=='__main__':
    app.run(host='0.0.0.0')