"""
This file predicts the output based on the
model, dataset, graph network and test data chosen by the user
"""
# import numpy as np
# import pandas as pd
# import sys
import os
# from random import shuffle
import torch
# import torch.nn as nn
from torch_geometric.data import DataLoader # pylint: disable=import-error
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils import *

MODELINGS = [GINConvNet, GATNet, GAT_GCN, GCNNet]
CUDA_NAME = "cuda:0" # pylint: disable=invalid-name
# cuda constant name uppercasing not possible

TEST_BATCH_SIZE = 512

def predicting(model, device, loader):
    """
    Loads the trained model and test data and does the prediction
    """
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print("Make prediction for {} samples...".format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0) # pylint: disable=no-member
            total_labels = torch.cat((total_labels, data.y_affinity.view(-1, 1).cpu()), 0) # pylint: disable=no-member
    return total_preds.numpy().flatten()

def predict(dataset, modeling):
    """
    Overall function used to ensure that the test data file
    exists and sets up the required files for the prediction
    """
    modeling = MODELINGS[modeling]
    processed_data_file_test = 'data/processed/test.pt'
    if not os.path.isfile(processed_data_file_test):
        print('please run create_data.py to prepare data in pytorch format!')
        return 0
    test_data = TestbedDataset(root='data', dataset='test')
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
    model_st = modeling.__name__
    print("\npredicting for ", dataset, " using ", model_st)
    # training the model
    device = torch.device(CUDA_NAME if torch.cuda.is_available() else "cpu") # pylint: disable=no-member
    model = modeling().to(device)
    model_file_name = "trained_models/model_" + model_st + "_" + dataset + ".model"
    if os.path.isfile(model_file_name):
        model.load_state_dict(torch.load(model_file_name, map_location=torch.device('cpu'))) # pylint: disable=no-member
        predicted_affinity = predicting(model, device, test_loader)
        print("The Predicted Drug Target Affinity is ", predicted_affinity)
        return predicted_affinity[0]
    print("model is not available!")
    return 0
