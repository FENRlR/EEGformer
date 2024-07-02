# Example of training script
import numpy as np
import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.preprocessing import StandardScaler
import models
import utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.manual_seed(1234)

print(f"CUDA : {torch.cuda.is_available()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# - Number of Channels : 1
esrinput = torch.tensor(pd.read_csv("./Epileptic Seizure Recognition/Epileptic Seizure Recognition.csv").values[0:, 1:178].astype(np.float32)) # truncation : 178 -> 177
esrlabel = torch.tensor(pd.read_csv("./Epileptic Seizure Recognition/Epileptic Seizure Recognition.csv").values[0:, 179].astype(np.float32)) # label stands at 179

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
sc = StandardScaler()
esrx = sc.fit_transform(esrx)
esrx = torch.tensor(esrx)
evalx = sc.transform(evalx)
evalx = torch.tensor(evalx)

print("Sample : ")
print(esrx)

print("Shapes : ")
print(esrx.shape)
print(evalx.shape)


# when the number of channels = 1
print(len(list(esrx.shape)))
if len(list(esrx.shape)) == 2:
    esrx = esrx.unsqueeze(2)
    evalx = evalx.unsqueeze(2)

print(esrx[0].shape)
print("-----------")

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
epoch = 30#100
bs = 750#500

keep_latest3 = True

load_pretrain = False

modelpath = ""
if load_pretrain is True:
    model = torch.load(modelpath)
else:
    model = models.EEGformer(esrx[0], num_cls, input_channels, kernel_size, num_blocks, num_heads_rtm, num_heads_stm, num_heads_ttm, num_submatrices, CF_second, dtype)
model.to(device)


num_data = esry.squeeze().shape[0]

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-04)  # 1e-05

preload = 0
if load_pretrain is True:
    preload = int(modelpath.split("/")[1].split(".")[0].split("_")[1])

log = []
logeval = []
#utils.initlossplot()
for i in range(epoch):
    log.append(0)
    logeval.append(0)
    model.train()
    for j in range((int)(num_data / bs)):
        optimizer.zero_grad()
        outputs = torch.zeros(bs, num_cls).to(device)
        label = esry[j * bs:j * bs + bs].to(dtype).to(device)
        #label = F.one_hot(label.long(), num_classes=num_cls) # one hot - for eegloss, eegloss_light, and eegloss_wol1

        for z in range(bs):
            inputs = esrx[j * bs + z].to(dtype).to(device)
            outputs[z] = model(inputs)

        #loss = model.eegloss(outputs, label, 0.005)# needs one hot encoding, L1_reg_const = 0.005
        loss = model.bceloss_w(outputs, label, truenum, esry.shape[0])
        loss.backward()
        optimizer.step()
        log[-1] += loss.item()
        print(f">>> b {j + 1} -> loss : {loss}")
    log[-1] = log[-1]/(int)(num_data / bs)
    #utils.lossplot_active(list(range(1,len(log)+1)),log)

    # Evaluation
    model.eval()
    with torch.no_grad():
        evoutputs = torch.zeros(evalx.shape[0]).to(device)
        tempoutputs = torch.zeros(evoutputs.shape[0], num_cls).to(device)
        evlabel = evaly.to(dtype).to(device)
        for z in range(evoutputs.shape[0]):
            evinputs = evalx[z].to(dtype).to(device)
            tempoutputs[z] = model(evinputs)
            evoutputs[z] = torch.argmax(tempoutputs[z].unsqueeze(0), dim=1)
        eval_loss = model.bceloss_w(tempoutputs, evlabel, truenum, esry.shape[0])
        logeval[-1] += eval_loss.item()

        tp, fp, tn, fn = utils.dscm(evoutputs, evlabel)
        print(f">>> epoch {i + 1} -> tp : {tp}, fp : {fp}, tn : {tn}, fn : {fn}")  # tp, fp, tn, fn

    torch.save(model, f'G_{preload + int(num_data / bs * (i + 1))}.pth')
    if keep_latest3 is True:
        delpath = f'./G_{preload + int(num_data / bs * (i - 2))}.pth'
        if os.path.exists(delpath):
            os.remove(delpath)

    print(f'saved : G_{preload + int(num_data / bs * (i + 1))}.pth')

#utils.lossplot(list(range(1,len(log)+1)),log)
utils.lossplot_with_val(list(range(1,len(log)+1)),log,logeval)