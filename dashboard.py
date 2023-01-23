import numpy as np 
import pandas as pd 

import streamlit as st 
import pickle
import requests
import shap
import json
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Modèle de Scoring",
    page_icon="🎈",
)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.write("""
         par *MAHAMADOU HAMIDOU Abdoul Aziz*
         """)

df = pd.read_csv('df_500.csv',index_col='SK_ID_CURR')
X = df.drop(columns=['TARGET'])
feature_names = X.columns

# Pickle
with open("classifier.pkl","rb") as pickle_in:
    classifier = pickle.load(pickle_in)
    
explainer = shap.TreeExplainer(classifier, X, feature_names=feature_names)
shap.initjs()

    
# explain model prediction shap results
def explain_model_prediction_shap(data):
    # Calculate Shap values
    shap_values = explainer(np.array(data))
    p = shap.plots.bar(shap_values)
    return p, shap_values 

def bivariate_analysis(feat1, feat_2,data):
    st.subheader('Bivariate Analysis')
    p = sns.scatterplot(data=data, x=data[feat1], y=data[feat_2], hue='TARGET',
                             color='red', s=100)
    return p
 
def plot_gauge(current_value, threshold):
    fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = current_value,
    title = {"text": "Current Value / Threshold Value"},
    gauge = {'axis': {'range': [0, 1]},
             'bar': {'color': "green"},
             'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': threshold}
            }))
    return fig
    
# get the data of the selected customer
def get_value(index):
    # Select the row at the specified index
    value = X.loc[index]
    return value.values.tolist()
  
def request_prediction(URI, data):
    
    response = requests.post(URI, json=data)
    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
    response = json.loads(response.text)
    response = pd.DataFrame(response)
    
    prediction = response['prediction'][0]
    probability = response['probability'][0]
    result = {'prediction':prediction, 'probability' : probability}
    return  result

def best_classification(probas, threshold, X):
    y_pred = 1 if probas > threshold else 0 
    return y_pred

def process():
    
    URI = 'http://scoringapi-env.eba-rrp7nmrb.eu-west-3.elasticbeanstalk.com:80/predict'
     
    st.title("Loan Default Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">loan payment risk prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
    Customer = st.sidebar.selectbox("Selectionner un numéro client: ",X.index)
    
    if st.sidebar.button("Predict"):
            data = get_value(Customer)
            result = request_prediction(URI, data)
            score = result['prediction']
            prob = result['probability']
            y_pred = best_classification(prob, 0.3918, data)
            if (y_pred == 1):
                risk_assessment = "Crédit refusé"
            else:
                risk_assessment = "Crédit accepté"
            st.sidebar.success(risk_assessment)
            st.sidebar.write("Probability: ", round(float(prob),4))
            st.sidebar.write(" best threshold: ", 0.3918)      
            st.subheader('Probability Gauge')
            gauge = plot_gauge(prob, 0.3918)  
            st.plotly_chart(gauge)
            st.subheader('Result Interpretability - Applicant Level')
            p, shap_values = explain_model_prediction_shap(data) 
            st.pyplot(p)
            st.subheader('Model Interpretability - Overall') 
            shap_values_ttl = explainer(X) 
            fig_ttl = shap.plots.bar(shap_values_ttl, max_display=10)
            st.pyplot(fig_ttl)
            st.pyplot(shap.summary_plot(shap.TreeExplainer(classifier).shap_values((X)), X, plot_type="bar"))
            
            
    
    
    selected_feature_1 = st.sidebar.selectbox('Feature 1', feature_names)
    selected_feature_2 = st.sidebar.selectbox('Feature 2', feature_names)
    if st.sidebar.button('display'):
                #data_chart = df.groupby("TARGET")[[selected_feature_1,selected_feature_2]].value_counts().unstack(level=0)
                #st.bar_chart(data_chart)
                p = bivariate_analysis(selected_feature_1, selected_feature_2, df)
                st.pyplot()
if __name__=='__main__':
    process() 