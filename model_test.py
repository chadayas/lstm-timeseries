import pandas as pd
import numpy as np
from sklearn import preprocessing
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import loadbond_data
import bond_forecasting as bf
import math


model = bf.model
testing_dataset = bf.test_dataset
crit = nn.MSELoss()

def testing(dataloader, model, loss_fn):
    model.eval()
    num_batches = len(dataloader)
    test_loss= 0
    
    with torch.no_grad():
        for x, y in dataloader:
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()     
        
    test_loss /= num_batches 
    print(f'Test Error: |Avg loss: {test_loss:>8f}|\n')   
epochs = 10

for t in range(epochs):
    print(f'epoch: {t+1}')
    print(f'-'*20)
    testing(testing_dataset,model,crit)
