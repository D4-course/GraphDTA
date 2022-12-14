"""
An overall file that will call the functions and run them
In future commits, the api will run here for the website
"""

from create_data import create_test
from predict_with_pretrained_model import predict as predict_with_pretrained_model
from fastapi import FastAPI
from pydantic import BaseModel

app  =  FastAPI()

class DTA(BaseModel):
    smiles: str
    protein: str
    dataset: str = 'davis'
    model: int = 0
    dta: float = 0.0

@app.post("/predict/")
def predict(dta: DTA):
    """
    This function will call the functions that will
    create the test data and then predict the affinity
    """
    create_test(dta.smiles, dta.protein)
    dta.dta = float(predict_with_pretrained_model(dta.dataset, dta.model))
    return dta
