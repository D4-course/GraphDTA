import streamlit as st
import requests

MODEL_MAPPING = {
    "GIN": 0,
    "GAT": 1,
    "GAT_GCN": 2,
    "GCN": 3
}

st.title('GraphDTA')

ml = st.form("Drug Target Binding Affinity Prediction")
drug = ml.text_input('Enter a drug SMILES')
protein = ml.text_input('Enter a protein sequence')
dataset = ml.selectbox("Choose the Dataset to use:", ("davis", "kiba"))
model = ml.selectbox("Choose the model to use:", ("GIN", "GAT", "GAT_GCN", "GCN"))
generate = ml.form_submit_button("Generate Binding Affinity")

if generate and (drug is not None) and (protein is not None) and (dataset is not None) and (model is not None):
    r = requests.post(url = "http://localhost:8000/predict/", json={"smiles": drug, "protein": protein, "dataset": dataset, "model": MODEL_MAPPING[model]})

    st.write("Binding Affinity: " + str(r.json()["dta"]))