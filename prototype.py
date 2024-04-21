import numpy as np
import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
import torchvision.ops as tos

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.manual_seed(1234)

print(f"쿠ㅡ다 : {torch.cuda.is_available()}")

# gpu utilization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ODCM(nn.Module):
    def __init__(self, input_channels, kernel_size, dtype= torch.float32):
        super(ODCM, self).__init__()
        #- 1D CNN
        # The outputs of the 1DCNN are defined as z3 ∈ R^(S×C×Le)
        # S : number of EEG channels = 20
        # C : number of depth-wise convolutional kernels used in the last layer
        # Le : features length outputted by the last layer
        """
        The input of the 1DCNN is an EEG segment represented using a two-dimensional (2D) matrix of size S × L, 
        where S represents the number of EEG channels, 
        and L represents the segment length. 
        
        The EEG segment is de-trend and normalized before being fed into the 1DCNN module, 
        and the normalized EEG segment is represented by x ∈ R^(S × L). 
        
        The 1DCNN adopts multiple depth-wise convolutions to extract EEG-channel-wise features and generate 3D feature maps. 
        
        It shifts across the data along the EEG channel dimension for each depth-wise convolution 
        and generates a 2D feature matrix of size S × Lf, 
        where Lf is the length of the extracted feature vector. 
        
        The output of the 1DCNN module is a 3D feature matrix of size S × C × Le, 
        where C is the number of depth-wise convolutional kernels used in the last layer of the 1DCNN module, 
        Le is the features length outputted by the last layer of the 1DCNN module.
        
        The size of the depth-wise convolutional filters used in the three layers is 1 × 10, 
        valid padding mode is applied in the three layers and the stride of the filters is set to 1.
         
        The number of the depth-wise convolutional filter used in the three layers is set to 120, 
        ensuring sufficient frequency features for learning the regional and synchronous characteristics. 
         
        We used a 3D coordinate system to depict the axis meaning of the 3D feature matrix. 
        The X, Y, and Z axes represent the temporal, spatial, 
        and convolutional feature information contained in the 3D feature matrix, respectively. 
         
        The output of the 1DCNN module is fed into the EEGformer encoder for encoding the EEG characteristics 
        (regional, temporal, and synchronous characteristics) in a unified manner. 
         
        The decoder is responsible for decoding the EEG characteristics 
        and inferencing the results according to the specific task.
        """

        self.inpch = input_channels
        self.ksize = kernel_size # 1X10
        self.dtype = dtype

        self.cvf1 = nn.Conv1d(in_channels = self.inpch, out_channels = self.inpch, kernel_size = self.ksize, padding='valid', stride=1, groups = self.inpch, dtype = self.dtype)
        self.cvf2 = nn.Conv1d(in_channels = self.cvf1.out_channels, out_channels = self.cvf1.out_channels, kernel_size = self.ksize, padding='valid', stride=1, groups = self.cvf1.out_channels, dtype = self.dtype)
        self.cvf3 = nn.Conv1d(in_channels = self.cvf2.out_channels, out_channels = 120 * self.cvf2.out_channels, kernel_size = self.ksize, padding='valid', stride=1, groups = self.cvf2.out_channels, dtype = self.dtype)

        #- reserve
        #self.relu1 = nn.ReLU()
        #self.relu2 = nn.ReLU()
        #self.relu3 = nn.ReLU()



    def forward(self, x):
        """
        The 1DCNN adopts multiple depth-wise convolutions to extract EEG-channel-wise features and generate 3D feature maps.
        It shifts across the data along the EEG channel dimension for each depth-wise convolution
        and generates a 2D feature matrix of size S × Lf, where Lf is the length of the extracted feature vector.
        The output of the 1DCNN module is a 3D feature matrix of size S × C × Le,
        where C is the number of depth-wise convolutional kernels used in the last layer of the 1DCNN module,
        Le is the features length outputted by the last layer of the 1DCNN module.

        More specifically, the 1DCNN is comprised of three depth-wise convolutional layers.
        Hence, we have the processing x → z1 → z2 → z3, where z1, z2, and z3 denote the outputs of the three layers.
        """
        print("ODCM forward")
        #x → z1
        x = self.cvf1(x)
        #x = self.relu1(x)
        print(x.shape)

        #z1 → z2
        x = self.cvf2(x)
        #x = self.relu2(x)
        print(x.shape)

        #z2 → z3
        x = self.cvf3(x)
        #x = self.relu3(x)
        print(x.shape)

        # S × L -> S × Lf (2D feature matrix) -> S × C × Le (3D feature matrix)
        # where C is the number of depth-wise convolutional kernels used in the 'last layer'
        x = torch.reshape(x, ((int)(x.shape[0]/(120)), 120, (int)(x.shape[1])))

        return x


class Mlp(nn.Module): # MLP from torchvision did not support float16 - borrowed from Meta
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., dtype=torch.float32):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, dtype = dtype)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, dtype = dtype)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RTM(nn.Module): # Regional transformer module
    def __init__(self, input, dtype): # input -> torch.Size([20, 120, 498]) #-> S x C x D
        super(RTM, self).__init__()
        self.inputshape = input.transpose(0, 1).transpose(1, 2).shape # - x : torch.Size([120, 498, 20]) - C x D x S
        self.M_size1 = self.inputshape[1] # -> D
        self.dtype = dtype

        self.weight = nn.Parameter(torch.randn(self.M_size1, self.inputshape[1], dtype = self.dtype)).to(device)
        self.bias = torch.zeros(self.inputshape[2], self.inputshape[0], self.M_size1, dtype = self.dtype).to(device)
        self.savespace = torch.zeros(self.inputshape[2], self.inputshape[0], self.M_size1, dtype = self.dtype).to(device) # S x C x D

        self.tK = 10 # number of transformer blocks - K in the paper
        self.hA = 1 # number of multi-head self-attention units (A is the number of units in a block)
        self.Dh = int(self.M_size1/self.hA) # Dh is the quotient computed by D/A and denotes the dimension number of three vectors.

        if self.M_size1%self.hA != 0 and int(self.M_size1/self.hA) == 0:
            print("ERROR 1")

        # Wq, Wk, Wv - the matrices of query, key, and value in the regional transformer module
        self.Wq = nn.Parameter(torch.randn((self.tK, self.hA, self.Dh, self.M_size1), dtype=self.dtype))
        self.Wk = nn.Parameter(torch.randn((self.tK, self.hA, self.Dh, self.M_size1), dtype=self.dtype))
        self.Wv = nn.Parameter(torch.randn((self.tK, self.hA, self.Dh, self.M_size1), dtype=self.dtype))

        self.Wo = nn.Parameter(torch.randn(self.tK, self.M_size1, self.M_size1, dtype=self.dtype)) # DxD -> DxDh in the paper but we gonna concatenate heads anyway and Dh*hA = D

        self.lnorm = nn.LayerNorm(self.M_size1, dtype=self.dtype) # LayerNorm operation for dimension D
        self.lnormz = nn.LayerNorm(self.M_size1, dtype=self.dtype)  # LayerNorm operation for z

        self.softmax = nn.Softmax()

        self.rsaspace = torch.zeros(self.inputshape[2], self.inputshape[0], self.tK, self.hA).to(device)# space for attention score


        #- Bias
        # "encode the position for each convolutional feature changing over time"
        # Each submatrix is represented by R^(C × Le)
        for i in range(self.inputshape[2]):  # (i = 1,2,3,…,S) -> S : number of EEG channels = 20
            for j in range(self.inputshape[0]):  # (j = 1,2,3,…,C) -> C : number of depth-wise convolutional kernels used in the last layer
                # The vector is sequentially taken out from the along the convolutional feature dimension and fed into the linear mapping module.
                for z in range(0,self.M_size1,2):
                    self.bias[i][j][z] = math.sin(j / 10000.0 ** (2 * z / self.M_size1))
                    self.bias[i][j][z + 1] = math.cos(j / 10000.0 ** (2 * z / self.M_size1))


        #self.mlp = tos.MLP(in_channels=self.M_size1, hidden_channels=[int(self.M_size1 * 4)], activation_layer=nn.GELU) # reserve - does not support float16
        self.mlp = Mlp(in_features=self.M_size1, hidden_features=int(self.M_size1 * 4), act_layer=nn.GELU, dtype=self.dtype) # mlp_ratio=4


    def forward(self, x):
        #  S x C x Le -> x
        x = x.transpose(0, 1).transpose(1, 2) # C x D x S
        print(f"- x : {x.shape}") # - x : torch.Size([120, 498, 20])- C x D x S
        #x = torch.matmul(self.weight, x) + self.bias

        for i in range(self.inputshape[2]):  # (i = 1,2,3,…,S) - i in the paper
            # The vector is sequentially taken out from the along the convolutional feature dimension and fed into the linear mapping module.
            # "sequentially taken out"
            for j in range(self.inputshape[0]):  # (j = 1,2,3,…,C) - c in the paper
                self.savespace[i][j] = torch.matmul(self.weight, x[j, :, i]) # R^(D)

        self.savespace = torch.add(self.savespace, self.bias) # z -> S x C x D

        #- Multi head attention
        # Q, K, V -> computed from the representation encoded by the preceding layer(l-1)
        qkvspace = torch.zeros(3, self.inputshape[2], self.inputshape[0], self.tK, self.hA, self.Dh).to(device)

        # intermediate vector space (s in paper) -> R^Dh
        imv = torch.zeros(self.inputshape[2], self.inputshape[0], self.tK, self.hA, self.Dh, dtype=self.dtype).to(device)

        for i in range(self.inputshape[2]):  # (i = 1,2,3,…,S) - i in the paper
            for j in range(self.inputshape[0]):  # (j = 1,2,3,…,C) - c in the paper
                for a in range(self.tK):  # blocks(layer)
                    for b in range(self.hA):  # heads per block
                        qkvspace[0, i, j, a, b] = torch.matmul( self.Wq[a,b], self.lnorm(self.savespace[i][j]) )  # Q
                        qkvspace[1, i, j, a, b] = torch.matmul( self.Wk[a,b], self.lnorm(self.savespace[i][j]) )  # K
                        qkvspace[2, i, j, a, b] = torch.matmul( self.Wv[a,b], self.lnorm(self.savespace[i][j]) )  # V

                        #- Attention score
                        self.rsaspace[i,j,a,b] = self.softmax( torch.matmul( (qkvspace[0,i,j,a,b]/math.sqrt(self.Dh)), qkvspace[1,i,j,a,b] ) )  # -> R^C (j에 대해 나열시?)

                        #- Intermediate vectors
                        # z renewed by l-th layer is computed by first concatenating the intermediate vectors from all heads,
                        # and the vector concatenation is projected by matrix Wo
                        #imv[i,j,a,b] = self.rsaspace[i,0,a,b]*qkvspace[2,i,0,a,b] # looks quite obsolete but anyway...

                        for subj in range(j):#0~j
                            imv[i,j,a,b] += self.rsaspace[i,subj,a,b]*qkvspace[2,i,subj,a,b]

                    #- NOW SAY HELLO TO NEW Z!
                    self.savespace[i,j] = self.Wo[a]@(imv[i,j,a,:].reshape(imv[i,j,a,:].shape[0]*imv[i,j,a,:].shape[1])) + self.savespace[i,j]# z' in the paper

                    #- normalized by LN() and passed through a multilayer perceptron (MLP)
                    #self.savespace[i, j] = self.mlp(self.lnormz(self.savespace[i, j]) + self.savespace[i,j]) # new z - reserve
                    self.savespace[i, j] = self.lnormz(self.savespace[i, j]) + self.savespace[i,j]
                    self.savespace[i, j] = self.mlp(self.savespace[i,j]) # new z

        return self.savespace # S x C x D - z4 in the paper - torch.Size([20, 120, 10])


class STM(nn.Module): # Synchronous transformer module
    def __init__(self, input, dtype): # input -> # S x C x D
        super(STM, self).__init__()
        self.inputshape = input.transpose(1, 2).shape # S x D x C (S x Le x C in the paper)
        self.M_size1 = self.inputshape[1] # -> D
        self.dtype = dtype

        self.weight = nn.Parameter(torch.randn(self.M_size1, self.inputshape[1], dtype = self.dtype)).to(device)
        self.bias = torch.zeros(self.inputshape[2], self.inputshape[0], self.M_size1, dtype = self.dtype).to(device)
        self.savespace = torch.zeros(self.inputshape[2], self.inputshape[0], self.M_size1, dtype = self.dtype).to(device) # C x S x D

        self.tK = 10 # number of transformer blocks - K in the paper
        self.hA = 1 # number of multi-head self-attention units (A is the number of units in a block)
        self.Dh = int(self.M_size1/self.hA) # Dh is the quotient computed by D/A and denotes the dimension number of three vectors.

        if self.M_size1%self.hA != 0 and int(self.M_size1/self.hA) == 0:
            print("ERROR 2")

        # Wq, Wk, Wv - the matrices of query, key, and value in the regional transformer module
        self.Wq = nn.Parameter(torch.randn((self.tK, self.hA, self.Dh, self.M_size1), dtype=self.dtype))
        self.Wk = nn.Parameter(torch.randn((self.tK, self.hA, self.Dh, self.M_size1), dtype=self.dtype))
        self.Wv = nn.Parameter(torch.randn((self.tK, self.hA, self.Dh, self.M_size1), dtype=self.dtype))

        self.Wo = nn.Parameter(torch.randn(self.tK, self.M_size1, self.M_size1, dtype=self.dtype)) # DxDh in the paper but we gonna concatenate heads anyway and Dh*hA = D

        self.lnorm = nn.LayerNorm(self.M_size1, dtype=self.dtype) # LayerNorm operation for dimension D
        self.lnormz = nn.LayerNorm(self.M_size1, dtype=self.dtype)  # LayerNorm operation for z

        self.softmax = nn.Softmax()

        self.rsaspace = torch.zeros(self.inputshape[2], self.inputshape[0], self.tK, self.hA).to(device)# space for attention score


        #- Bias
        for i in range(self.inputshape[2]):  # (i = 1,2,3,…,C)
            for j in range(self.inputshape[0]):  # (j = 1,2,3,…,S)
                # The vector is sequentially taken out from the along the convolutional feature dimension and fed into the linear mapping module.
                for z in range(0,self.M_size1,2):
                    self.bias[i][j][z] = math.sin(j / 10000.0 ** (2 * z / self.M_size1))
                    self.bias[i][j][z + 1] = math.cos(j / 10000.0 ** (2 * z / self.M_size1))


        #self.mlp = tos.MLP(in_channels=self.M_size1, hidden_channels=[int(self.M_size1 * 4)], activation_layer=nn.GELU) # reserve - does not support float16
        self.mlp = Mlp(in_features=self.M_size1, hidden_features=int(self.M_size1 * 4), act_layer=nn.GELU, dtype=self.dtype) # mlp_ratio=4


    def forward(self, x):
        # S x C x D -> x
        x = x.transpose(1, 2) # S x D x C
        print(f"- x : {x.shape}") # - x : torch.Size([120, 20, 10])
        #x = torch.matmul(self.weight, x) + self.bias

        for i in range(self.inputshape[2]):  # (i = 1,2,3,…,C)
            for j in range(self.inputshape[0]):  # (j = 1,2,3,…,S)
                self.savespace[i][j] = torch.matmul(self.weight, x[j, :, i]) # R^(D)

        self.savespace = torch.add(self.savespace, self.bias) # z -> C x S x D

        #- Multi head attention
        # Q, K, V -> computed from the representation encoded by the preceding layer(l-1)
        qkvspace = torch.zeros(3, self.inputshape[2], self.inputshape[0], self.tK, self.hA, self.Dh).to(device)

        # intermediate vector space (s in paper) -> R^Dh
        imv = torch.zeros(self.inputshape[2], self.inputshape[0], self.tK, self.hA, self.Dh, dtype=self.dtype).to(device)

        for i in range(self.inputshape[2]):  # (i = 1,2,3,…,C) - i in the paper
            for j in range(self.inputshape[0]):  # (j = 1,2,3,…,S) - s in the paper
                for a in range(self.tK):  # blocks(layer)
                    for b in range(self.hA):  # heads per block
                        qkvspace[0, i, j, a, b] = torch.matmul( self.Wq[a,b], self.lnorm(self.savespace[i][j]) )  # Q
                        qkvspace[1, i, j, a, b] = torch.matmul( self.Wk[a,b], self.lnorm(self.savespace[i][j]) )  # K
                        qkvspace[2, i, j, a, b] = torch.matmul( self.Wv[a,b], self.lnorm(self.savespace[i][j]) )  # V

                        #- Attention score
                        self.rsaspace[i,j,a,b] = self.softmax( torch.matmul( (qkvspace[0,i,j,a,b]/math.sqrt(self.Dh)), qkvspace[1,i,j,a,b] ) )

                        #- Intermediate vectors
                        # z renewed by l-th layer is computed by first concatenating the intermediate vectors from all heads,
                        # and the vector concatenation is projected by matrix Wo
                        #imv[i,j,a,b] = self.rsaspace[i,0,a,b]*qkvspace[2,i,0,a,b] # looks quite obsolete but anyway...

                        for subj in range(j):#0~j
                            imv[i,j,a,b] += self.rsaspace[i,subj,a,b]*qkvspace[2,i,subj,a,b]

                    #- NOW SAY HELLO TO NEW Z!
                    self.savespace[i,j] = self.Wo[a]@(imv[i,j,a,:].reshape(imv[i,j,a,:].shape[0]*imv[i,j,a,:].shape[1])) + self.savespace[i,j]# z' in the paper

                    #- Normalized by LN() and passed through a multilayer perceptron (MLP)
                    #self.savespace[i, j] = self.mlp(self.lnormz(self.savespace[i, j]) + self.savespace[i,j]) # new z - reserve
                    self.savespace[i, j] = self.lnormz(self.savespace[i, j]) + self.savespace[i,j]
                    self.savespace[i, j] = self.mlp(self.savespace[i,j]) # new z

        return self.savespace # C x S x D - z5 in the paper - torch.Size([120, 20, 10])


class TTM(nn.Module): # Temporal transformer module
    def __init__(self, input, avgf, dtype): # input -> # C x S x D
        super(TTM, self).__init__()
        """
        To avoid huge computational complexity, we compress the original temporal dimensionality D of z5 into dimensionality M.
        That is, the 3D matrix z5 is first segmented and then averaged into M 2D submatrices along the temporal dimension.
        Each submatrix is represented by Xi ∈ R^(S×C)(i = 1,2,3,…,M) and the M submatrices are concatenated to form X ∈ R^(M×S×C).
        """
        # Xtemp -> R^(M x S x C)
        # Xtemp-i -> R^(S x C)
        # Xtemp'-i -> R^L1, where L1 = S x C (flatten)
        # Xtemp' -> R^(M x L1)
        self.dtype = dtype

        #-Compress D to M, where M = avgf
        self.avgf = avgf # average factor (M)
        self.input = input.transpose(0, 2) # D x S x C
        self.seg = self.input.shape[0]/self.avgf
        if self.input.shape[0]%self.avgf != 0 and int(self.input.shape[0]/self.avgf) == 0:
            print("ERROR 3")

        self.inputc = torch.zeros(self.avgf, self.input.shape[1], self.input.shape[2], dtype=self.dtype).to(device) # M x S x C
        for i in range(0, self.avgf): # each i consists self.input.shape[0]/avgf
            for j in range(int(i*self.seg), int((i+1)*self.seg)):
                self.inputc[i, :, :] += self.input[j, :, :]
            self.inputc[i, :, :] = self.inputc[i, :, :]/self.seg

        #- Flattened into a vector
        self.inputcf = self.inputc.reshape(self.avgf, self.input.shape[1]*self.input.shape[2])# M x L1 -> M x (S*C)
        self.inputshape = self.inputcf.shape # M x L1
        #self.M_size1 = M_size1 # M_size1 = self.eD -> D
        self.M_size1 = self.inputshape[1] # L1

        self.weight = nn.Parameter(torch.randn(self.M_size1, self.inputshape[1], dtype = self.dtype)).to(device) # D x L
        self.bias = torch.zeros(self.inputshape[0], self.M_size1, dtype = self.dtype).to(device) # ei ∈ R^D -> e ∈ M x D
        self.savespace = torch.zeros(self.inputshape[0], self.M_size1, dtype = self.dtype).to(device) # M x D

        self.tK = 10 # number of transformer blocks - K in the paper
        self.hA = 1 # number of multi-head self-attention units (A is the number of units in a block)
        self.Dh = int(self.M_size1/self.hA) # Dh is the quotient computed by D/A and denotes the dimension number of three vectors.

        if self.M_size1%self.hA != 0 and int(self.M_size1/self.hA) == 0:
            print("ERROR 4")

        # Wq, Wk, Wv - the matrices of query, key, and value in the regional transformer module
        self.Wq = nn.Parameter(torch.randn((self.tK, self.hA, self.Dh, self.M_size1), dtype=self.dtype))
        self.Wk = nn.Parameter(torch.randn((self.tK, self.hA, self.Dh, self.M_size1), dtype=self.dtype))
        self.Wv = nn.Parameter(torch.randn((self.tK, self.hA, self.Dh, self.M_size1), dtype=self.dtype))

        self.Wo = nn.Parameter(torch.randn(self.tK, self.M_size1, self.M_size1, dtype=self.dtype)) # L1xL1 -> L1xL in the paper, where L*hA = L1

        self.lnorm = nn.LayerNorm(self.M_size1, dtype=self.dtype) # LayerNorm operation for dimension D
        self.lnormz = nn.LayerNorm(self.M_size1, dtype=self.dtype)  # LayerNorm operation for z

        self.softmax = nn.Softmax()

        self.rsaspace = torch.zeros(self.inputshape[0], self.tK, self.hA).to(device)# space for attention score


        #- Bias
        # M x L -> shape of the input
        for i in range(self.inputshape[0]):  # (i = 1,2,3,…,M)
            for j in range(0,self.M_size1,2):# (j = 1,2,3,…,D)
                    self.bias[i][j] = math.sin(i / 10000.0 ** (2 * j / self.M_size1))
                    self.bias[i][j + 1] = math.cos(i / 10000.0 ** (2 * j / self.M_size1))


        #self.mlp = tos.MLP(in_channels=self.M_size1, hidden_channels=[int(self.M_size1 * 4)], activation_layer=nn.GELU) # reserve - does not support float16
        self.mlp = Mlp(in_features=self.M_size1, hidden_features=int(self.M_size1 * 4), act_layer=nn.GELU, dtype=self.dtype) # mlp_ratio=4


    def redefine(self, input):
        #- compress D to M, where M = avgf
        input = input.transpose(0, 2)  # D x S x C
        inputc = torch.zeros(self.avgf, input.shape[1], input.shape[2], dtype=self.dtype).to(device)  # M x S x C
        for i in range(0, self.avgf):  # each i consists self.input.shape[0]/avgf
            for j in range(int(i*self.seg), int((i+1)*self.seg)): # int(i*self.seg), int((i+1)*self.seg)
                inputc[i, :, :] += input[j, :, :]
            inputc[i, :, :] = inputc[i, :, :] / self.seg

        #- flattened into a vector
        inputcf = inputc.reshape(self.avgf, input.shape[1] * input.shape[2])  # M x L -> M x (S*C)

        return inputcf


    def forward(self, x):
        #  C x S x D -> x
        x = self.redefine(x) # M x L -> M x (S*C)
        #print(f"- x : {x.shape}") # - x : torch.Size([2, 2400])
        #x = torch.matmul(self.weight, x) + self.bias

        for i in range(self.inputshape[0]): # (i = 1,2,3,…,M)
            self.savespace[i] = torch.matmul(self.weight, x[i, :]) # R^(D)

        self.savespace = torch.add(self.savespace, self.bias) # z -> M x D

        #- Multi head attention
        # Q, K, V -> computed from the representation encoded by the preceding layer(l-1)
        qkvspace = torch.zeros(3, self.inputshape[0], self.tK, self.hA, self.Dh).to(device)

        #- Intermediate vector space (s in paper) -> R^Dh
        imv = torch.zeros(self.inputshape[0], self.tK, self.hA, self.Dh, dtype=self.dtype).to(device)

        for i in range(self.inputshape[0]):  # (i = 1,2,3,…,M) - i in the paper
            for a in range(self.tK):  # blocks(layer)
                for b in range(self.hA):  # heads per block
                    qkvspace[0, i, a, b] = torch.matmul(self.Wq[a, b], self.lnorm(self.savespace[i]))  # Q
                    qkvspace[1, i, a, b] = torch.matmul(self.Wk[a, b], self.lnorm(self.savespace[i]))  # K
                    qkvspace[2, i, a, b] = torch.matmul(self.Wv[a, b], self.lnorm(self.savespace[i]))  # V

                    # - Attention score
                    self.rsaspace[i, a, b] = self.softmax(torch.matmul((qkvspace[0, i, a, b] / math.sqrt(self.Dh)), qkvspace[1, i, a, b]))

                    #- Intermediate vectors
                    # z renewed by l-th layer is computed by first concatenating the intermediate vectors from all heads,
                    # and the vector concatenation is projected by matrix Wo
                    #imv[i, a, b] = self.rsaspace[0, a, b] * qkvspace[2, 0, a, b]  # looks quite obsolete but anyway...

                    for subj in range(i):
                        imv[i, a, b] += self.rsaspace[subj, a, b] * qkvspace[2, subj, a, b]

                #- NOW SAY HELLO TO NEW Z!
                self.savespace[i] = self.Wo[a] @ (imv[i, a, :].reshape(imv[i, a, :].shape[0] * imv[i, a, :].shape[1])) + self.savespace[i]  # z' in the paper - # M x D

                #- Normalized by LN() and passed through a multilayer perceptron (MLP)
                # self.savespace[i, j] = self.mlp(self.lnormz(self.savespace[i, j]) + self.savespace[i,j]) # new z - reserve
                self.savespace[i] = self.lnormz(self.savespace[i]) + self.savespace[i]
                self.savespace[i] = self.mlp(self.savespace[i])  # new z

        return self.savespace.reshape(self.avgf, self.input.shape[1], self.input.shape[2]) # M x S x C - O in the paper which is M x (S*C) but we need M x S x C for the decoder anyway


class CNNdecoder(nn.Module): # EEGformer decoder
    def __init__(self, input, N, dtype): # input -> # M x S x C
        super(CNNdecoder, self).__init__()
        """
        The EEGformer is used to extract the temporal, regional, and synchronous characteristics in a unified manner, 
        as well as to deal with various EEG-based brain activity analysis tasks. Unlike the original transformer decoder, 
        which uses a multi-head self-attention mechanism to decode the feature output of the corresponding encoder, 
        we designed a convolution neural network (CNN) to perform the corresponding task. 
        The CNN contains three convolutional layers and one fully connected layer.
        """
        self.input = input.transpose(0, 1).transpose(1, 2)  # S x C x M
        self.s = self.input.shape[0]  # S
        self.c = self.input.shape[1]  # C
        self.m = self.input.shape[2]  # M
        self.n = N # N denotes the number of convolutional filters used in the second layer.

        self.dtype = dtype

        # The CNN contains three convolutional layers and one fully connected layer.
        # The first layer of our EEGformer decoder linearly combined different convolutional features
        # for normalization across the convolutional dimension.
        # x -> torch.Size([20, 120, 10]) -> S x C x M
        # W1 -> C x 1
        self.cvd1 = nn.Conv1d(in_channels=self.c, out_channels=1, kernel_size=1, dtype=self.dtype) # S x M
        print(f"W1 -> C x 1 : {self.cvd1.weight.shape}") # 즉 가중치의 크기는 torch.Size([out_channels, in_channels, kernel_size])

        # W2 -> S x N (where N denotes the number of convolutional filters used in the second layer)
        self.cvd2 = nn.Conv1d(in_channels=self.s, out_channels=self.n, kernel_size=1, dtype=self.dtype)# M x N
        print(f"W2 -> S x N : {self.cvd2.weight.shape}")

        # W3 -> M/2 x M (M/2 x N in the paper, but I think they've mistyped it since the dimension of the output is defined as M/2 x N)
        self.cvd3 = nn.Conv1d(in_channels=self.m, out_channels=int(self.m/2), kernel_size=1, dtype=self.dtype) # M/2 x N
        print(f"W3 -> M/2 x N : {self.cvd3.weight.shape}")

        # The fourth layer of our CNN is a fully connected layer that produced classification results for the brain activity analysis task.
        self.fc = nn.Linear(int(self.m/2)*self.n, 1, dtype=self.dtype)
        print(f"Wfc : {self.fc.weight.shape}")



        # Loss = (1/Dn) * sigma(i=1~Dn){-log(Pi(Yi)) + lambda * abs(w)}
        # where Dn is the number of data samples in the training dataset,
        # Pi and Yi are the prediction results produced by the model and the corresponding ground truth label for the i-th data sample,
        # and λ is the constant of the L1 regularization. (λ > 0, is manually tuned)


    def forward(self, x): # x -> M x S x C
        x = x.transpose(0,1).transpose(1,2) # S x C x M
        print(x.shape) # torch.Size([20, 120, 10])

        x = self.cvd1(x).squeeze()
        print(x.shape) # torch.Size([20, 10]) -> S x M
        print("shape of conv x1")

        x = self.cvd2(x).transpose(0,1).squeeze() # N x M transposed to M x N
        print(x.shape)  # torch.Size([10, 2]) -> M x N
        print("shape of conv x2")

        x = self.cvd3(x).squeeze()
        print(x.shape)  # torch.Size([5, 2]) -> M/2 x N
        print("shape of conv x3")

        x = self.fc(x.reshape(1, x.shape[0]*x.shape[1]))#.squeeze()

        return x




#- EXPERIMENTAL

# Parameters
input_channels = 20
kernel_size = 10
#dtype = torch.float16
dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("-- RAW DATA")
reade = pd.read_excel("./dummydata/exeeg1.xlsx")
inputs = torch.tensor(reade.values[0:, 0:]).to(dtype).to(device)
#print(inputs)
print(inputs.shape) #torch.Size([525, 20])


print("-- Preprocess")
model = ODCM(input_channels, kernel_size, dtype)
model.to(device)

outputs = model(torch.transpose(inputs,0,1))
#outputs = model(inputs)
print("- output : Preprocess")
#print(outputs)
print(outputs.shape) # torch.Size([20, 120, 498])


print("-- RTM")
model2 = RTM(outputs, dtype)
model2.to(device)

outputs2 = model2(outputs)

print("- output : RTM")
#print(outputs2)
print(outputs2.shape)# S x C x D - torch.Size([20, 120, 10])


print("-- STM")
model3 = STM(outputs2, dtype)
model3.to(device)

outputs3 = model3(outputs2)

print("- output : STM")
#print(outputs3)
print(outputs3.shape) # torch.Size([120, 20, 10])
#"""


#outputs3 = torch.randn([120, 20, 10],dtype=dtype).to(device)# dummy
print("-- TTM")
model4 = TTM(outputs3, 10, dtype)
model4.to(device)

outputs4 = model4(outputs3)

print("- output : TTM")
#print(outputs4)
print(outputs4.shape) #torch.Size([10, 20, 120])


print("-- CNNdecoder")
model5 = CNNdecoder(outputs4, 2, dtype)
model5.to(device)

outputs5 = model5(outputs4)

print("- output : CNNdecoder")
print(outputs5)
print(outputs5.shape)

"""
-- RAW DATA
torch.Size([525, 20])

-- Preprocess
- output : Preprocess
torch.Size([20, 120, 498])

- output : RTM
torch.Size([20, 120, 498])

- output : STM
torch.Size([120, 20, 498])

- output : TTM
torch.Size([10, 20, 120])

- output : CNNdecoder
tensor([[62.8309]], device='cuda:0', grad_fn=<AddmmBackward0>)
torch.Size([1, 1])
"""