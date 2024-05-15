# Conceptual demonstration of training script. - Not for actual use.
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.manual_seed(1234)

print(f"CUDA : {torch.cuda.is_available()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# - Number of Channels : 1
esrinput = torch.tensor(pd.read_csv("./Epileptic Seizure Recognition/Epileptic Seizure Recognition.csv").values[0:, 1:178].astype(np.float32))
esrlabel = torch.tensor(pd.read_csv("./Epileptic Seizure Recognition/Epileptic Seizure Recognition.csv").values[0:, 178].astype(np.float32))

# label에서 1이 아닌 경우 모조리 0으로 분류
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

# scale
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
esrx = sc.fit_transform(esrx)  # AttributeError: 'numpy.ndarray' object has no attribute 'unsqueeze'
esrx = torch.tensor(esrx)
evalx = sc.transform(evalx)
evalx = torch.tensor(evalx)

print("sample : ")
print(esrx)

print("shapes : ")
print(esrx.shape)
print(evalx.shape)


# when number of channels = 1
print(len(list(esrx.shape)))
if len(list(esrx.shape)) == 2:
    esrx = esrx.unsqueeze(2)
    evalx = evalx.unsqueeze(2)

print(esrx[0].shape)


# Parameters
input_channels = esrx.shape[2]
num_cls = 2
kernel_size = 10
num_blocks = 3  # number of transformer blocks - K in the paper

# num_head divides input.shape[0]-3*(self.kernel_size-1)
num_heads_rtm = 6  # number of multi-head self-attention units (A is the number of units in a block)
num_heads_stm = 6
num_heads_ttm = 11

num_submatrices = 10
CF_second = 2

# dtype = torch.float16
dtype = torch.float32
epoch = 5#100
bs = 750  # 500

load_pretrain = True#False

modelpath = "./G_70.pth"
if load_pretrain is True:
    model = torch.load(modelpath)
else:
    model = models.EEGformer(esrx[0], num_cls, input_channels, kernel_size, num_blocks, num_heads_rtm, num_heads_stm, num_heads_ttm, num_submatrices, CF_second, dtype)
model.to(device)


num_data = esry.squeeze().shape[0]

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-03)  # 1e-05


def dscm(x, y):
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(x.shape[0]):
        if x[i] == 1:
            if x[i] == y[i]:
                tp += 1
            else:
                fp += 1
        else:
            if x[i] == y[i]:
                tn += 1
            else:
                fn += 1

    # acc = (tp+tn)/(tp + fp + tn + fn)
    # sen = tp/(tp+fn)
    # spe = tn/(tn+fp) # ZeroDivisionError
    if (tp + fp + tn + fn) != 0:
        print(f"acc = {(tp + tn) / (tp + fp + tn + fn)}")
    else:
        print("ERROR - acc : (tp + fp + tn + fn) = 0")
    if (tp + fn) != 0:
        print(f"sen = {tp / (tp + fn)}")
    else:
        print("ERROR - sen : (tp+fn) = 0")
    if (tn + fp) != 0:
        print(f"spe = {tn / (tn + fp)}")
    else:
        print("ERROR - spe : (tn+fp) = 0")

    return tp, fp, tn, fn
    # return acc, sen, spe


preload = 0
if load_pretrain is True:
    preload = int(modelpath.split("/")[1].split(".")[0].split("_")[1])

for i in range(epoch):
    for j in range((int)(num_data / bs)):
        optimizer.zero_grad()
        outputs = torch.zeros(bs, num_cls).to(device)
        label = esry[j * bs:j * bs + bs].to(dtype).to(device)
        for z in range(bs):
            inputs = esrx[j * bs + z].to(dtype).to(device)
            outputs[z] = model(inputs)

        # loss = model.eegloss(outputs, label, L1_reg_const = 0.00005)#L1_reg_const = 0.005
        loss = model.eegloss_w(outputs, label, 0.00005, truenum, esry.shape[0])  # L1_reg_const = 0.005
        loss.backward()
        optimizer.step()
        print(f">>> bs {j + 1} -> loss : {loss}")

    # Evaluation
    with torch.no_grad():
        evoutputs = torch.zeros(evalx.shape[0]).to(device)
        evlabel = evaly.to(dtype).to(device)
        for z in range(evalx.shape[0]):
            evinputs = evalx[z].to(dtype).to(device)
            evoutputs[z] = torch.round(torch.max(model(evinputs)))

        # acc,sen,spe = dscm(evoutputs,evlabel)
        tp, fp, tn, fn = dscm(evoutputs, evlabel)
        # print(f">>> epoch {i+1} -> ACC : {acc}, SEN : {sen}, SPE : {spe}")
        print(f">>> epoch {i + 1} -> tp : {tp}, fp : {fp}, tn : {tn}, fn : {fn}")  # tp, fp, tn, fn

    torch.save(model, f'G_{preload + int(num_data / bs * (i + 1))}.pth')
    print(f'saved : G_{preload + int(num_data / bs * (i + 1))}.pth')