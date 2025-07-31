import streamlit as st 
import pandas as pd
import numpy as np
import pickle 
from sklearn.preprocessing import LabelEncoder,StandardScaler

def load_model():
    with open("student_lr_final_model.pkl","rb")as file:
        model,scaler,lb=pickle.load(file)
        return model,scaler,lb
def preprocesing_input_data(data,scaler,lb):
    data["Extracurricular Activities"]=lb.transform([data["Extracurricular Activities"]])
    df=pd.DataFrame([data])
    df_transformed=scaler.transform(df)
    return df_transformed

def predict_data(data):
    model,scaler,le=load_model()
    processed_data=preprocesing_input_data(data,scaler,le)
    prediction=model.predict(processed_data)
    return prediction

def main():
    st.title("student performance prediction")
    st.write("enter your data to get a prediction for your performance")
    
    hour_studied=st.number_input("hours studied",min_value=1,max_value=10,value=5)
    prvious_score=st.number_input("previous score",min_value=40,max_value=100,value=70)
    extra=st.selectbox("extra curri activity",['Yes',"No"])
    sleeping_hour=st.number_input("sleeping hours",min_value=4,max_value=10,value=7)
    number_of_paper_solved=st.number_input("number of question paper solved",min_value=0,max_value=10,value=5)
    
    if st.button("predict_your_score"):
        user_data={
            "Hours Studied":hour_studied,
            "Previous Scores": prvious_score,
            "Extracurricular Activities":extra,
            "Sleep Hours": sleeping_hour,
            "Sample Question Papers Practiced": number_of_paper_solved
        }
        prediction=predict_data(user_data)
        st.success(f"your prediction result is {prediction}")
if __name__=="__main__":
    main()
    

    
    