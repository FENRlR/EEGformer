import os
import torch
import torch.nn as nn
import math
import torchvision.ops as tos


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ODCM(nn.Module): # 1D CNN Module
    def __init__(self, input_channels, kernel_size, dtype=torch.float32):
        super(ODCM, self).__init__()
        """
        The input of the 1DCNN is an EEG segment represented using a two-dimensional (2D) matrix of size S × L, 
        where S represents the number of EEG channels, and L represents the segment length. 
        The 1DCNN adopts multiple depth-wise convolutions to extract EEG-channel-wise features and generate 3D feature maps. 
        It shifts across the data along the EEG channel dimension for each depth-wise convolution 
        and generates a 2D feature matrix of size S × Lf, where Lf is the length of the extracted feature vector. 
        The output of the 1DCNN module is a 3D feature matrix of size S × C × Le, 
        where C is the number of depth-wise convolutional kernels used in the last layer of the 1DCNN module, 
        Le is the features length outputted by the last layer of the 1DCNN module.
        """
        self.inpch = input_channels
        self.ksize = kernel_size  # The size of the depth-wise convolutional filters used in the three layers is 1 × 10
        self.ncf = 120  # The number of the depth-wise convolutional filter used in the three layers is set to 120
        self.dtype = dtype

        self.cvf1 = nn.Conv1d(in_channels=self.inpch, out_channels=self.inpch, kernel_size=self.ksize, padding='valid', stride=1, groups=self.inpch, dtype=self.dtype)
        self.cvf2 = nn.Conv1d(in_channels=self.cvf1.out_channels, out_channels=self.cvf1.out_channels, kernel_size=self.ksize, padding='valid', stride=1, groups=self.cvf1.out_channels, dtype=self.dtype)
        self.cvf3 = nn.Conv1d(in_channels=self.cvf2.out_channels, out_channels=self.ncf * self.cvf2.out_channels, kernel_size=self.ksize, padding='valid', stride=1, groups=self.cvf2.out_channels, dtype=self.dtype)

        # - reserve
        # self.relu1 = nn.ReLU()
        # self.relu2 = nn.ReLU()
        # self.relu3 = nn.ReLU()

    def forward(self, x):
        # x → z1
        x = self.cvf1(x)
        # x = self.relu1(x)

        # z1 → z2
        x = self.cvf2(x)
        # x = self.relu2(x)

        # z2 → z3
        x = self.cvf3(x)
        # x = self.relu3(x)

        # S × L -> S × Lf (2D feature matrix) -> S × C × Le (3D feature matrix)
        # where C is the number of depth-wise convolutional kernels used in the last layer
        x = torch.reshape(x, ((int)(x.shape[0] / self.ncf), self.ncf, (int)(x.shape[1])))

        return x


class Mlp(nn.Module):  # MLP from torchvision did not support float16 - borrowed from Meta
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., dtype=torch.float32):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, dtype=dtype)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, dtype=dtype)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RTM(nn.Module):  # Regional transformer module
    def __init__(self, input, num_blocks, num_heads, dtype):  # input -> S x C x D
        super(RTM, self).__init__()
        self.inputshape = input.transpose(0, 1).transpose(1, 2).shape  # C x D x S
        self.M_size1 = self.inputshape[1]  # -> D
        self.dtype = dtype

        self.weight = nn.Parameter(torch.randn(self.M_size1, self.inputshape[1], dtype=self.dtype))
        self.bias = torch.zeros(self.inputshape[2], self.inputshape[0], self.M_size1, dtype=self.dtype).to(device)
        self.savespace = torch.zeros(self.inputshape[2], self.inputshape[0], self.M_size1, dtype=self.dtype).to(device) # S x C x D

        self.tK = num_blocks  # number of transformer blocks - K in the paper
        self.hA = num_heads  # number of multi-head self-attention units (A is the number of units in a block)
        self.Dh = int(self.M_size1 / self.hA)  # Dh is the quotient computed by D/A and denotes the dimension number of three vectors.

        if self.M_size1 % self.hA != 0 and int(self.M_size1 / self.hA) == 0:
            print("ERROR 1")

        # Wq, Wk, Wv - the matrices of query, key, and value in the regional transformer module
        self.Wq = nn.Parameter(torch.randn((self.tK, self.hA, self.Dh, self.M_size1), dtype=self.dtype))
        self.Wk = nn.Parameter(torch.randn((self.tK, self.hA, self.Dh, self.M_size1), dtype=self.dtype))
        self.Wv = nn.Parameter(torch.randn((self.tK, self.hA, self.Dh, self.M_size1), dtype=self.dtype))

        self.Wo = nn.Parameter(torch.randn(self.tK, self.M_size1, self.M_size1, dtype=self.dtype))   # DxD -> DxDh in the paper but we gonna concatenate heads anyway and Dh*hA = D

        self.lnorm = nn.LayerNorm(self.M_size1, dtype=self.dtype)  # LayerNorm operation for dimension D
        self.lnormz = nn.LayerNorm(self.M_size1, dtype=self.dtype)  # LayerNorm operation for z

        self.softmax = nn.Softmax()

        self.rsaspace = torch.zeros(self.inputshape[2], self.inputshape[0], self.tK, self.hA, dtype=self.dtype).to(device)   # space for attention score
        self.qkvspace = torch.zeros(3, self.inputshape[2], self.inputshape[0], self.tK, self.hA, self.Dh, dtype=self.dtype).to(device) # Q, K, V

        # intermediate vector space (s in paper) -> R^Dh
        self.imv = torch.zeros(self.inputshape[2], self.inputshape[0], self.tK, self.hA, self.Dh, dtype=self.dtype).to(device)

        # - Bias
        # "encode the position for each convolutional feature changing over time"
        # Each submatrix is represented by R^(C × Le)
        for i in range(self.inputshape[2]):  # (i = 1,2,3,…,S) -> S : number of EEG channels = 20
            for j in range(self.inputshape[0]):  # (j = 1,2,3,…,C) -> C : number of depth-wise convolutional kernels used in the last layer
                # The vector is sequentially taken out from the along the convolutional feature dimension and fed into the linear mapping module.
                for z in range(0, self.M_size1, 2):
                    self.bias[i][j][z] = math.sin(j / 10000.0 ** (2 * z / self.M_size1))
                    self.bias[i][j][z + 1] = math.cos(j / 10000.0 ** (2 * z / self.M_size1))

        # self.mlp = tos.MLP(in_channels=self.M_size1, hidden_channels=[int(self.M_size1 * 4)], activation_layer=nn.GELU) # reserve - does not support float16
        self.mlp = Mlp(in_features=self.M_size1, hidden_features=int(self.M_size1 * 4), act_layer=nn.GELU, dtype=self.dtype) # mlp_ratio=4

    def forward(self, x):
        #  S x C x Le -> x
        x = x.transpose(0, 1).transpose(1, 2)  # C x D x S

        for i in range(self.inputshape[2]):  # (i = 1,2,3,…,S) - i in the paper
            for j in range(self.inputshape[0]):  # (j = 1,2,3,…,C) - c in the paper
                self.savespace[i][j] = torch.matmul(self.weight, x[j, :, i])  # R^(D)

        self.savespace = torch.add(self.savespace, self.bias)  # z -> S x C x D

        # - Multi head attention
        # Q, K, V -> computed from the representation encoded by the preceding layer(l-1)
        for i in range(self.inputshape[2]):  # (i = 1,2,3,…,S) - i in the paper
            for j in range(self.inputshape[0]):  # (j = 1,2,3,…,C) - c in the paper
                for a in range(self.tK):  # blocks(layer)
                    for b in range(self.hA):  # heads per block
                        self.qkvspace[0, i, j, a, b] = torch.matmul(self.Wq[a, b], self.lnorm(self.savespace[i][j]))  # Q
                        self.qkvspace[1, i, j, a, b] = torch.matmul(self.Wk[a, b], self.lnorm(self.savespace[i][j]))  # K
                        self.qkvspace[2, i, j, a, b] = torch.matmul(self.Wv[a, b], self.lnorm(self.savespace[i][j]))  # V

                        # - Attention score
                        self.rsaspace[i, j, a, b] = self.softmax(torch.matmul((self.qkvspace[0, i, j, a, b] / math.sqrt(self.Dh)),self.qkvspace[1, i, j, a, b]))  # -> R^C (j에 대해 나열시?)

                        # - Intermediate vectors
                        # z renewed by l-th layer is computed by first concatenating the intermediate vectors from all heads,
                        # and the vector concatenation is projected by matrix Wo
                        self.imv[i, j, a, b] = self.rsaspace[i, 0, a, b] * self.qkvspace[2, i, 0, a, b]  # looks quite obsolete but anyway...

                        for subj in range(1, j + 1):  # 0~j
                            self.imv[i, j, a, b] += self.rsaspace[i, subj, a, b] * self.qkvspace[2, i, subj, a, b]

                    # - NOW SAY HELLO TO NEW Z!
                    self.savespace[i, j] = self.Wo[a] @ (self.imv[i, j, a, :].reshape(self.imv[i, j, a, :].shape[0] * self.imv[i, j, a, :].shape[1])) + self.savespace[i, j]  # z' in the paper

                    # - normalized by LN() and passed through a multilayer perceptron (MLP)
                    # self.savespace[i, j] = self.mlp(self.lnormz(self.savespace[i, j]) + self.savespace[i,j]) # new z - reserve
                    self.savespace[i, j] = self.lnormz(self.savespace[i, j]) + self.savespace[i, j]
                    self.savespace[i, j] = self.mlp(self.savespace[i, j])  # new z

        return self.savespace  # S x C x D - z4 in the paper


class STM(nn.Module):  # Synchronous transformer module
    def __init__(self, input, num_blocks, num_heads, dtype):  # input -> # S x C x D
        super(STM, self).__init__()
        self.inputshape = input.transpose(1, 2).shape  # S x D x C (S x Le x C in the paper)
        self.M_size1 = self.inputshape[1]  # -> D
        self.dtype = dtype

        self.weight = nn.Parameter(torch.randn(self.M_size1, self.inputshape[1], dtype=self.dtype))
        self.bias = torch.zeros(self.inputshape[2], self.inputshape[0], self.M_size1, dtype=self.dtype).to(device)
        self.savespace = torch.zeros(self.inputshape[2], self.inputshape[0], self.M_size1, dtype=self.dtype).to(device)   # C x S x D

        self.tK = num_blocks  # number of transformer blocks - K in the paper
        self.hA = num_heads  # number of multi-head self-attention units (A is the number of units in a block)
        self.Dh = int(self.M_size1 / self.hA)  # Dh is the quotient computed by D/A and denotes the dimension number of three vectors.

        if self.M_size1 % self.hA != 0 and int(self.M_size1 / self.hA) == 0:
            print("ERROR 2")

        # Wq, Wk, Wv - the matrices of query, key, and value in the synchronous transformer module
        self.Wq = nn.Parameter(torch.randn((self.tK, self.hA, self.Dh, self.M_size1), dtype=self.dtype))
        self.Wk = nn.Parameter(torch.randn((self.tK, self.hA, self.Dh, self.M_size1), dtype=self.dtype))
        self.Wv = nn.Parameter(torch.randn((self.tK, self.hA, self.Dh, self.M_size1), dtype=self.dtype))

        self.Wo = nn.Parameter(torch.randn(self.tK, self.M_size1, self.M_size1, dtype=self.dtype)) # DxDh in the paper but we gonna concatenate heads anyway and Dh*hA = D

        self.lnorm = nn.LayerNorm(self.M_size1, dtype=self.dtype)  # LayerNorm operation for dimension D
        self.lnormz = nn.LayerNorm(self.M_size1, dtype=self.dtype)  # LayerNorm operation for z

        self.softmax = nn.Softmax()

        self.rsaspace = torch.zeros(self.inputshape[2], self.inputshape[0], self.tK, self.hA, dtype=self.dtype).to(device)  # space for attention score
        self.qkvspace = torch.zeros(3, self.inputshape[2], self.inputshape[0], self.tK, self.hA, self.Dh, dtype=self.dtype).to(device) # Q, K, V

        # intermediate vector space (s in paper) -> R^Dh
        self.imv = torch.zeros(self.inputshape[2], self.inputshape[0], self.tK, self.hA, self.Dh, dtype=self.dtype).to(device)

        # - Bias
        for i in range(self.inputshape[2]):  # (i = 1,2,3,…,C)
            for j in range(self.inputshape[0]):  # (j = 1,2,3,…,S)
                # The vector is sequentially taken out from the along the convolutional feature dimension and fed into the linear mapping module.
                for z in range(0, self.M_size1, 2):
                    self.bias[i][j][z] = math.sin(j / 10000.0 ** (2 * z / self.M_size1))
                    self.bias[i][j][z + 1] = math.cos(j / 10000.0 ** (2 * z / self.M_size1))

        # self.mlp = tos.MLP(in_channels=self.M_size1, hidden_channels=[int(self.M_size1 * 4)], activation_layer=nn.GELU) # reserve - does not support float16
        self.mlp = Mlp(in_features=self.M_size1, hidden_features=int(self.M_size1 * 4), act_layer=nn.GELU, dtype=self.dtype) # mlp_ratio=4

    def forward(self, x):
        # S x C x D -> x
        x = x.transpose(1, 2)  # S x D x C

        for i in range(self.inputshape[2]):  # (i = 1,2,3,…,C)
            for j in range(self.inputshape[0]):  # (j = 1,2,3,…,S)
                self.savespace[i][j] = torch.matmul(self.weight, x[j, :, i])  # R^(D)

        self.savespace = torch.add(self.savespace, self.bias)  # z -> C x S x D

        # - Multi head attention
        # Q, K, V -> computed from the representation encoded by the preceding layer(l-1)
        for i in range(self.inputshape[2]):  # (i = 1,2,3,…,C) - i in the paper
            for j in range(self.inputshape[0]):  # (j = 1,2,3,…,S) - s in the paper
                for a in range(self.tK):  # blocks(layer)
                    for b in range(self.hA):  # heads per block
                        self.qkvspace[0, i, j, a, b] = torch.matmul(self.Wq[a, b], self.lnorm(self.savespace[i][j]))  # Q
                        self.qkvspace[1, i, j, a, b] = torch.matmul(self.Wk[a, b], self.lnorm(self.savespace[i][j]))  # K
                        self.qkvspace[2, i, j, a, b] = torch.matmul(self.Wv[a, b], self.lnorm(self.savespace[i][j]))  # V

                        # - Attention score
                        self.rsaspace[i, j, a, b] = self.softmax(torch.matmul((self.qkvspace[0, i, j, a, b] / math.sqrt(self.Dh)), self.qkvspace[1, i, j, a, b]))

                        # - Intermediate vectors
                        # z renewed by l-th layer is computed by first concatenating the intermediate vectors from all heads,
                        # and the vector concatenation is projected by matrix Wo
                        self.imv[i, j, a, b] = self.rsaspace[i, 0, a, b] * self.qkvspace[2, i, 0, a, b]  # looks quite obsolete but anyway...

                        for subj in range(1, j + 1):  # 0~j
                            self.imv[i, j, a, b] += self.rsaspace[i, subj, a, b] * self.qkvspace[2, i, subj, a, b]

                    # - NOW SAY HELLO TO NEW Z!
                    self.savespace[i, j] = self.Wo[a] @ (self.imv[i, j, a, :].reshape(self.imv[i, j, a, :].shape[0] * self.imv[i, j, a, :].shape[1])) + self.savespace[i, j]  # z' in the paper

                    # - Normalized by LN() and passed through a multilayer perceptron (MLP)
                    # self.savespace[i, j] = self.mlp(self.lnormz(self.savespace[i, j]) + self.savespace[i,j]) # new z - reserve
                    self.savespace[i, j] = self.lnormz(self.savespace[i, j]) + self.savespace[i, j]
                    self.savespace[i, j] = self.mlp(self.savespace[i, j])  # new z

        return self.savespace  # C x S x D - z5 in the paper


class TTM(nn.Module):  # Temporal transformer module
    def __init__(self, input, num_submatrices, num_blocks, num_heads, dtype):  # input -> # C x S x D
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

        # -Compress D to M, where M = avgf
        self.avgf = num_submatrices  # average factor (M)
        self.input = input.transpose(0, 2)  # D x S x C
        self.seg = self.input.shape[0] / self.avgf
        if self.input.shape[0] % self.avgf != 0 and int(self.input.shape[0] / self.avgf) == 0:
            print("ERROR 3")

        self.inputc = torch.zeros(self.avgf, self.input.shape[1], self.input.shape[2], dtype=self.dtype).to(device)  # M x S x C
        for i in range(0, self.avgf):  # each i consists self.input.shape[0]/avgf
            for j in range(int(i * self.seg), int((i + 1) * self.seg)):
                self.inputc[i, :, :] += self.input[j, :, :]
            self.inputc[i, :, :] = self.inputc[i, :, :] / self.seg

        # - Flattened into a vector
        self.inputcf = self.inputc.reshape(self.avgf, self.input.shape[1] * self.input.shape[2])  # M x L1 -> M x (S*C)
        self.inputshape = self.inputcf.shape  # M x L1
        # self.M_size1 = M_size1 # M_size1 = self.eD -> D
        self.M_size1 = self.inputshape[1]  # L1

        self.weight = nn.Parameter(torch.randn(self.M_size1, self.inputshape[1], dtype=self.dtype)) # D x L
        self.bias = torch.zeros(self.inputshape[0], self.M_size1, dtype=self.dtype).to(device) # ei ∈ R^D -> e ∈ M x D
        self.savespace = torch.zeros(self.inputshape[0], self.M_size1, dtype=self.dtype).to(device) # M x D

        self.tK = num_blocks  # number of transformer blocks - K in the paper
        self.hA = num_heads  # number of multi-head self-attention units (A is the number of units in a block)
        self.Dh = int(self.M_size1 / self.hA)  # Dh is the quotient computed by D/A and denotes the dimension number of three vectors.

        if self.M_size1 % self.hA != 0 and int(self.M_size1 / self.hA) == 0:
            print("ERROR 4")

        # Wq, Wk, Wv - the matrices of query, key, and value in the temporal transformer module
        self.Wq = nn.Parameter(torch.randn((self.tK, self.hA, self.Dh, self.M_size1), dtype=self.dtype))
        self.Wk = nn.Parameter(torch.randn((self.tK, self.hA, self.Dh, self.M_size1), dtype=self.dtype))
        self.Wv = nn.Parameter(torch.randn((self.tK, self.hA, self.Dh, self.M_size1), dtype=self.dtype))

        self.Wo = nn.Parameter(torch.randn(self.tK, self.M_size1, self.M_size1, dtype=self.dtype))  # L1xL1 -> L1xL in the paper, where L*hA = L1

        self.lnorm = nn.LayerNorm(self.M_size1, dtype=self.dtype)  # LayerNorm operation for dimension D
        self.lnormz = nn.LayerNorm(self.M_size1, dtype=self.dtype)  # LayerNorm operation for z

        self.softmax = nn.Softmax()

        self.rsaspace = torch.zeros(self.inputshape[0], self.tK, self.hA, dtype=self.dtype).to(device) # space for attention score
        self.qkvspace = torch.zeros(3, self.inputshape[0], self.tK, self.hA, self.Dh, dtype=self.dtype).to(device)# Q, K, V

        # - Intermediate vector space (s in paper) -> R^Dh
        self.imv = torch.zeros(self.inputshape[0], self.tK, self.hA, self.Dh, dtype=self.dtype).to(device)

        # - Bias
        # M x L -> shape of the input
        for i in range(self.inputshape[0]):  # (i = 1,2,3,…,M)
            for j in range(0, self.M_size1, 2):  # (j = 1,2,3,…,D)
                self.bias[i][j] = math.sin(i / 10000.0 ** (2 * j / self.M_size1))
                self.bias[i][j + 1] = math.cos(i / 10000.0 ** (2 * j / self.M_size1))

        # self.mlp = tos.MLP(in_channels=self.M_size1, hidden_channels=[int(self.M_size1 * 4)], activation_layer=nn.GELU) # reserve - does not support float16
        self.mlp = Mlp(in_features=self.M_size1, hidden_features=int(self.M_size1 * 4), act_layer=nn.GELU, dtype=self.dtype)  # mlp_ratio=4

    def redefine(self, input):
        # - compress D to M, where M = avgf
        input = input.transpose(0, 2)  # D x S x C
        inputc = torch.zeros(self.avgf, input.shape[1], input.shape[2], dtype=self.dtype).to(device) # M x S x C
        for i in range(0, self.avgf):  # each i consists self.input.shape[0]/avgf
            for j in range(int(i * self.seg), int((i + 1) * self.seg)):  # int(i*self.seg), int((i+1)*self.seg)
                inputc[i, :, :] += input[j, :, :]
            inputc[i, :, :] = inputc[i, :, :] / self.seg

        # - flattened into a vector
        inputcf = inputc.reshape(self.avgf, input.shape[1] * input.shape[2])  # M x L -> M x (S*C)

        return inputcf

    def forward(self, x):
        #  C x S x D -> x
        x = self.redefine(x)  # M x L -> M x (S*C)

        for i in range(self.inputshape[0]):  # (i = 1,2,3,…,M)
            self.savespace[i] = torch.matmul(self.weight, x[i, :])  # R^(D)

        self.savespace = torch.add(self.savespace, self.bias)  # z -> M x D

        # - Multi head attention
        # Q, K, V -> computed from the representation encoded by the preceding layer(l-1)
        for i in range(self.inputshape[0]):  # (i = 1,2,3,…,M) - i in the paper
            for a in range(self.tK):  # blocks(layer)
                for b in range(self.hA):  # heads per block
                    self.qkvspace[0, i, a, b] = torch.matmul(self.Wq[a, b], self.lnorm(self.savespace[i]))  # Q
                    self.qkvspace[1, i, a, b] = torch.matmul(self.Wk[a, b], self.lnorm(self.savespace[i]))  # K
                    self.qkvspace[2, i, a, b] = torch.matmul(self.Wv[a, b], self.lnorm(self.savespace[i]))  # V

                    # - Attention score
                    self.rsaspace[i, a, b] = self.softmax(torch.matmul((self.qkvspace[0, i, a, b] / math.sqrt(self.Dh)), self.qkvspace[1, i, a, b]))

                    # - Intermediate vectors
                    # z renewed by l-th layer is computed by first concatenating the intermediate vectors from all heads,
                    # and the vector concatenation is projected by matrix Wo
                    self.imv[i, a, b] = self.rsaspace[0, a, b] * self.qkvspace[2, 0, a, b]  # looks quite obsolete but anyway...

                    for subj in range(1, i + 1):
                        self.imv[i, a, b] += self.rsaspace[subj, a, b] * self.qkvspace[2, subj, a, b]

                # - NOW SAY HELLO TO NEW Z!
                self.savespace[i] = self.Wo[a] @ (self.imv[i, a, :].reshape(self.imv[i, a, :].shape[0] * self.imv[i, a, :].shape[1])) + self.savespace[i]  # z' in the paper - # M x D

                # - Normalized by LN() and passed through a multilayer perceptron (MLP)
                # self.savespace[i, j] = self.mlp(self.lnormz(self.savespace[i, j]) + self.savespace[i,j]) # new z - reserve
                self.savespace[i] = self.lnormz(self.savespace[i]) + self.savespace[i]
                self.savespace[i] = self.mlp(self.savespace[i])  # new z

        return self.savespace.reshape(self.avgf, self.input.shape[1], self.input.shape[2])  # M x S x C - O in the paper which is M x (S*C) but we need M x S x C for the decoder anyway


class CNNdecoder(nn.Module):  # EEGformer decoder
    def __init__(self, input, CF_second, dtype):  # input -> # M x S x C
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
        self.n = CF_second  # N denotes the number of convolutional filters used in the second layer.

        self.dtype = dtype

        # The CNN contains three convolutional layers and one fully connected layer.
        # The first layer of our EEGformer decoder linearly combined different convolutional features
        # for normalization across the convolutional dimension.

        # W1 -> C x 1
        self.cvd1 = nn.Conv1d(in_channels=self.c, out_channels=1, kernel_size=1, dtype=self.dtype)  # S x M

        # W2 -> S x N (where N denotes the number of convolutional filters used in the second layer)
        self.cvd2 = nn.Conv1d(in_channels=self.s, out_channels=self.n, kernel_size=1, dtype=self.dtype)  # N x M -> will be transposed to M x N later

        # W3 -> M/2 x M (M/2 x N in the paper, but I think they've mistyped it since the dimension of the output is defined as M/2 x N)
        self.cvd3 = nn.Conv1d(in_channels=self.m, out_channels=int(self.m / 2), kernel_size=1, dtype=self.dtype)  # M/2 x N

        # The fourth layer of our CNN is a fully connected layer that produced classification results for the brain activity analysis task.
        self.fc = nn.Linear(int(self.m / 2) * self.n, 1, dtype=self.dtype)

    def forward(self, x):  # x -> M x S x C
        x = x.transpose(0, 1).transpose(1, 2)  # S x C x M
        x = self.cvd1(x).squeeze()  # S x M
        x = self.cvd2(x).transpose(0, 1).squeeze()  # N x M transposed to M x N
        x = self.cvd3(x).squeeze()  # M/2 x N
        x = self.fc(x.reshape(1, x.shape[0] * x.shape[1]))

        return x


class EEGformer(nn.Module):
    def __init__(self, input, input_channels, kernel_size, num_blocks, num_heads, num_submatrices, CF_second, dtype=torch.float32):
        super(EEGformer, self).__init__()
        self.dtype = dtype
        self.ncf = 120
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.tK = num_blocks
        self.hA = num_heads
        self.avgf = num_submatrices
        self.cfs = CF_second

        self.outshape1 = torch.zeros(self.input_channels, self.ncf, input.shape[0]-3*(self.kernel_size-1)).to(device) # torch.Size([525, 20])  -> torch.Size([20, 120, 498])
        self.outshape2 = torch.zeros(self.outshape1.shape[0], self.outshape1.shape[1], self.outshape1.shape[2]).to(device)# S x C x D - torch.Size([20, 120, 498])
        self.outshape3 = torch.zeros(self.outshape2.shape[1], self.outshape2.shape[0], self.outshape2.shape[2]).to(device) # torch.Size([120, 20, 498])
        self.outshape4 = torch.zeros(self.avgf, self.outshape3.shape[1], self.outshape3.shape[0]).to(device) # torch.Size([10, 20, 120])

        self.odcm = ODCM(input_channels, self.kernel_size, self.dtype)
        self.rtm = RTM(self.outshape1, self.tK, self.hA, self.dtype)
        self.stm = STM(self.outshape2,  self.tK, self.hA, self.dtype)
        self.ttm = TTM(self.outshape3, self.avgf,  self.tK, self.hA, self.dtype)
        self.cnndecoder = CNNdecoder(self.outshape4, self.cfs, self.dtype)

    def forward(self, x):
        x = self.odcm(x.transpose(0, 1))
        x = self.rtm(x)
        x = self.stm(x)
        x = self.ttm(x)
        x = self.cnndecoder(x)
        return x


class eegloss(nn.Module):  # Loss function
    def __init__(self, L1_reg_const, w):
        super(eegloss, self).__init__()
        """
        Loss = (1/Dn) * sigma(i=1~Dn){-log(Pi(Yi)) + λ * abs(w)}
        where Dn is the number of data samples in the training dataset,
        Pi and Yi are the prediction results produced by the model and the corresponding ground truth label for the i-th data sample,
        and λ is the constant of the L1 regularization. (λ > 0, is manually tuned)
        w is differentiable everywhere except when w = 0
        """
        self.lrc = L1_reg_const
        self.w = nn.Parameter(torch.tensor(w))

    def forward(self, x, label):
        x = torch.mean(-torch.log(x @ label) + self.lrc * torch.abs(self.w))
        return x
