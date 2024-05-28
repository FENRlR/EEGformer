# Example of Inference
import numpy as np
import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
from sklearn.preprocessing import StandardScaler
import models
import utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.manual_seed(1234)

print(f"CUDA : {torch.cuda.is_available()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# - Number of Channels : 1
esrinput = torch.tensor(pd.read_csv("./Epileptic seizure recognition/Epileptic Seizure Recognition.csv").values[0:, 1:178].astype(np.float32))
esrlabel = torch.tensor(pd.read_csv("./Epileptic seizure recognition/Epileptic Seizure Recognition.csv").values[0:, 178].astype(np.float32))

for i in range(esrlabel.shape[0]):
    if esrlabel[i] != 1:
        esrlabel[i] = 0

# train dataset
esrx = esrinput[0:10500, :]
esry = esrlabel[0:10500]

truenum = 0
for i in range(esry.shape[0]):
    if esry[i] == 1:
        truenum += 1

# test dataset
evalx = esrinput[10500:11500, :]
evaly = esrlabel[10500:11500]

num_data = esry.squeeze().shape[0]

# scale
sc = StandardScaler()
esrx = sc.fit_transform(esrx)  # AttributeError: 'numpy.ndarray' object has no attribute 'unsqueeze'
esrx = torch.tensor(esrx)
evalx = sc.transform(evalx)
evalx = torch.tensor(evalx)


if len(list(esrx.shape)) == 2:
    esrx = esrx.unsqueeze(2)
    evalx = evalx.unsqueeze(2)


# dtype = torch.float16
dtype = torch.float32


modelpath = ""
print(f">> {modelpath.split('/')[1]}")
model = torch.load(modelpath)
model.to(device)
model.eval()


with torch.no_grad():
    evoutputs = torch.zeros(evalx.shape[0]).to(device)
    evlabel = evaly.to(dtype).to(device)
    #evoutputs = torch.zeros(esrx.shape[0]).to(device)
    #evlabel = esry.to(dtype).to(device)
    for z in range(evoutputs.shape[0]):
        evinputs = esrx[z].to(dtype).to(device)
        evoutputs[z] = torch.argmax(model(evinputs), dim=1)

    tp, fp, tn, fn = utils.dscm(evoutputs, evlabel)
    print(f"tp : {tp}, fp : {fp}, tn : {tn}, fn : {fn}")  # tp, fp, tn, fn
