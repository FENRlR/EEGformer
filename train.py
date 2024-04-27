# Conceptual demonstration of training script. - Not for actual use.
import numpy as np
import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
import torchvision.ops as tos
import models

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.manual_seed(1234)

print(f"CUDA : {torch.cuda.is_available()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
input_channels = 20
kernel_size = 10
num_blocks = 10 # number of transformer blocks - K in the paper
num_heads = 1 # number of multi-head self-attention units (A is the number of units in a block)
num_submatrices = 10 # number of submatrices in temporal transformer module
CF_second = 2 # # the number of convolutional filters used in the second layer in decoder - N in the paper

# dtype = torch.float16
dtype = torch.float32

epoch = 1


inputs = torch.tensor(pd.read_excel("./dummydata/exeeg1.xlsx").values[0:, 0:]).to(dtype).to(device)  # torch.Size([5, 20])
print(inputs)
print(inputs.shape)

model = models.EEGformer(inputs, input_channels, kernel_size, num_blocks, num_heads, num_submatrices, CF_second, dtype)
model.to(device)

inputs = None

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

label = torch.tensor(pd.read_excel(f"./dummydata/labels.xlsx").values[0:, 0:]).to(dtype).to(device)
num_data = label.squeeze().shape[0]
lossf = models.eegloss(L1_reg_const = 1.0, w = 0.5)


dummyval_i = torch.tensor(pd.read_excel("./dummydata/exeeg1.xlsx").values[0:, 0:]).to(dtype).to(device)
dummyval_l = torch.tensor(pd.read_excel(f"./dummydata/labels.xlsx").values[0:, 0:]).to(dtype).to(device)


for i in range(epoch):
    outputs = torch.zeros(num_data).to(device)

    for j in range(1, num_data + 1):
        inputs = torch.tensor(pd.read_excel(f"./dummydata/exeeg{j}.xlsx").values[0:, 0:]).to(dtype).to(device)
        #label = torch.tensor(pd.read_excel(f"./dummydata/labels.xlsx").values[0:, 0:]).to(dtype).to(device)

        outputs[j - 1] = model(inputs)
        print(f"output : {outputs[j - 1]}")

    loss = lossf(outputs, label)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    print(f">>> epoch {i+1} -> loss : {loss}")