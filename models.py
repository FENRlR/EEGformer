import os
import torch
import torch.nn as nn
import math
import torchvision.ops as tos


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ODCM(nn.Module): # 1D CNN Module
    def __init__(self, input_channels, kernel_size, dtype=torch.float32):
        super(ODCM, self).__init__()

        self.inpch = input_channels
        self.ksize = kernel_size  # 1X10
        self.ncf = 120  # The number of the depth-wise convolutional filter used in the three layers is set to 120
        self.dtype = dtype

        self.cvf1 = nn.Conv1d(in_channels=self.inpch, out_channels=self.inpch, kernel_size=self.ksize, padding='valid', stride=1, groups=self.inpch, dtype=self.dtype)
        self.cvf2 = nn.Conv1d(in_channels=self.cvf1.out_channels, out_channels=self.cvf1.out_channels, kernel_size=self.ksize, padding='valid', stride=1, groups=self.cvf1.out_channels, dtype=self.dtype)
        self.cvf3 = nn.Conv1d(in_channels=self.cvf2.out_channels, out_channels=self.ncf * self.cvf2.out_channels, kernel_size=self.ksize, padding='valid', stride=1, groups=self.cvf2.out_channels, dtype=self.dtype)

    def forward(self, x):
        x = self.cvf1(x)
        x = self.cvf2(x)
        x = self.cvf3(x)
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
        self.savespace = torch.zeros(self.inputshape[2], self.inputshape[0], self.M_size1, dtype=self.dtype).to(device)  # S x C x D

        self.tK = num_blocks  # number of transformer blocks - K in the paper
        self.hA = num_heads  # number of multi-head self-attention units (A is the number of units in a block)
        self.Dh = int(self.M_size1 / self.hA)  # Dh is the quotient computed by D/A and denotes the dimension number of three vectors.

        if self.M_size1 % self.hA != 0 and int(self.M_size1 / self.hA) == 0:
            print("ERROR 1")

        self.Wqkv = nn.Parameter(torch.randn((self.tK, 3, self.hA, self.Dh, self.M_size1), dtype=self.dtype))

        self.Wo = nn.Parameter(torch.randn(self.tK, self.M_size1, self.M_size1, dtype=self.dtype))
        self.lnorm = nn.LayerNorm(self.M_size1, dtype=self.dtype)  # LayerNorm operation for dimension D
        self.lnormz = nn.LayerNorm(self.M_size1, dtype=self.dtype)  # LayerNorm operation for z
        self.softmax = nn.Softmax()
        self.rsaspace = torch.zeros(self.tK, self.inputshape[2], self.inputshape[0], self.hA, dtype=self.dtype).to(device)  # space for attention score
        self.qkvspace = torch.zeros(self.tK, 3, self.inputshape[2], self.inputshape[0], self.hA, self.Dh, dtype=self.dtype).to(device)  # Q, K, V
        self.imv = torch.zeros(self.tK, self.inputshape[2], self.inputshape[0], self.hA, self.Dh, dtype=self.dtype).to(device)

        self.mlp = Mlp(in_features=self.M_size1, hidden_features=int(self.M_size1 * 4), act_layer=nn.GELU, dtype=self.dtype)  # mlp_ratio=4

    def forward(self, x):
        x = x.transpose(0, 1).transpose(1, 2)  # C x D x S

        self.savespace = torch.einsum('lm,jmi -> ijl', self.weight, x)

        # - Bias
        for i in range(self.inputshape[2]):  # (i = 1,2,3,…,S) -> S : number of EEG channels = 20
            for j in range(self.inputshape[0]):  # (j = 1,2,3,…,C) -> C : number of depth-wise convolutional kernels used in the last layer
                # The vector is sequentially taken out from the along the convolutional feature dimension and fed into the linear mapping module.
                for z in range(0, self.M_size1, 2):
                    self.bias[i][j][z] = math.sin(j / 10000.0 ** (2 * z / self.M_size1))
                    self.bias[i][j][z + 1] = math.cos(j / 10000.0 ** (2 * z / self.M_size1))

        self.savespace = torch.add(self.savespace, self.bias)  # z -> S x C x D

        for a in range(self.tK):  # blocks(layer)
            self.qkvspace[a] = torch.einsum('xhdm,ijm -> xijhd', self.Wqkv[a], self.lnorm(self.savespace))  # Q, K, V

            # - Attention score
            self.rsaspace[a] = torch.einsum('ijhd,ijhd -> ijh', self.qkvspace[a, 0].clone() / math.sqrt(self.Dh), self.qkvspace[a, 1].clone())

            # - Intermediate vectors
            self.imv[a] = torch.einsum('ijh,ijhd -> ijhd', self.rsaspace[a].clone(), self.qkvspace[a, 2].clone())

            for subj in range(1, self.inputshape[0]):
                self.imv[a, :, subj] = self.imv[a, :, subj] + self.imv[a, :, subj - 1]

            # - NOW SAY HELLO TO NEW Z!
            self.savespace = torch.einsum('nm,ijm -> ijn', self.Wo[a], self.imv.clone().reshape(self.tK, self.inputshape[2], self.inputshape[0], self.M_size1)[a]) + self.savespace  # z'

            # - normalized by LN() and passed through a multilayer perceptron (MLP)
            self.savespace = self.lnormz(self.savespace) + self.savespace
            self.savespace = self.mlp(self.savespace)  # new z

        return self.savespace  # S x C x D - z4 in the paper


class STM(nn.Module):  # Synchronous transformer module
    def __init__(self, input, num_blocks, num_heads, dtype):  # input -> # S x C x D
        super(STM, self).__init__()
        self.inputshape = input.transpose(1, 2).shape  # S x D x C (S x Le x C in the paper)
        self.M_size1 = self.inputshape[1]  # -> D
        self.dtype = dtype
        self.weight = nn.Parameter(torch.randn(self.M_size1, self.inputshape[1], dtype=self.dtype))
        self.bias = torch.zeros(self.inputshape[2], self.inputshape[0], self.M_size1, dtype=self.dtype).to(device)
        self.savespace = torch.zeros(self.inputshape[2], self.inputshape[0], self.M_size1, dtype=self.dtype).to(device)  # C x S x D
        self.tK = num_blocks  # number of transformer blocks - K in the paper
        self.hA = num_heads  # number of multi-head self-attention units (A is the number of units in a block)
        self.Dh = int(self.M_size1 / self.hA)  # Dh is the quotient computed by D/A and denotes the dimension number of three vectors.

        if self.M_size1 % self.hA != 0 and int(self.M_size1 / self.hA) == 0:
            print("ERROR 2")

        self.Wqkv = nn.Parameter(torch.randn((self.tK, 3, self.hA, self.Dh, self.M_size1), dtype=self.dtype))
        self.Wo = nn.Parameter(torch.randn(self.tK, self.M_size1, self.M_size1, dtype=self.dtype))  # DxDh in the paper but we gonna concatenate heads anyway and Dh*hA = D
        self.lnorm = nn.LayerNorm(self.M_size1, dtype=self.dtype)  # LayerNorm operation for dimension D
        self.lnormz = nn.LayerNorm(self.M_size1, dtype=self.dtype)  # LayerNorm operation for z
        self.softmax = nn.Softmax()
        self.rsaspace = torch.zeros(self.tK, self.inputshape[2], self.inputshape[0], self.hA, dtype=self.dtype).to(device)  # space for attention score
        self.qkvspace = torch.zeros(self.tK, 3, self.inputshape[2], self.inputshape[0], self.hA, self.Dh, dtype=self.dtype).to(device)  # Q, K, V
        self.imv = torch.zeros(self.tK, self.inputshape[2], self.inputshape[0], self.hA, self.Dh, dtype=self.dtype).to(device)

        self.mlp = Mlp(in_features=self.M_size1, hidden_features=int(self.M_size1 * 4), act_layer=nn.GELU, dtype=self.dtype)  # mlp_ratio=4

    def forward(self, x): # S x C x D
        x = x.transpose(1, 2)  # S x D x C

        self.savespace = torch.einsum('lm,jmi -> ijl', self.weight, x)

        # - Bias
        for i in range(self.inputshape[2]):  # (i = 1,2,3,…,C)
            for j in range(self.inputshape[0]):  # (j = 1,2,3,…,S)
                # The vector is sequentially taken out from the along the convolutional feature dimension and fed into the linear mapping module.
                for z in range(0, self.M_size1, 2):
                    self.bias[i][j][z] = math.sin(j / 10000.0 ** (2 * z / self.M_size1))
                    self.bias[i][j][z + 1] = math.cos(j / 10000.0 ** (2 * z / self.M_size1))

        self.savespace = torch.add(self.savespace, self.bias)  # z -> C x S x D

        for a in range(self.tK):  # blocks(layer)
            self.qkvspace[a] = torch.einsum('xhdm,ijm -> xijhd', self.Wqkv[a], self.lnorm(self.savespace))  # Q, K, V

            # - Attention score
            self.rsaspace[a] = torch.einsum('ijhd,ijhd -> ijh', self.qkvspace[a, 0].clone() / math.sqrt(self.Dh), self.qkvspace[a, 1].clone())

            # - Intermediate vectors
            self.imv[a] = torch.einsum('ijh,ijhd -> ijhd', self.rsaspace[a].clone(), self.qkvspace[a, 2].clone())

            for subj in range(1, self.inputshape[0]):
                self.imv[a, :, subj] = self.imv[a, :, subj] + self.imv[a, :, subj - 1]

            # - NOW SAY HELLO TO NEW Z!
            self.savespace = torch.einsum('nm,ijm -> ijn', self.Wo[a], self.imv.clone().reshape(self.tK, self.inputshape[2], self.inputshape[0], self.M_size1)[a]) + self.savespace  # z'

            # - normalized by LN() and passed through a multilayer perceptron (MLP)
            self.savespace = self.lnormz(self.savespace) + self.savespace
            self.savespace = self.mlp(self.savespace)  # new z

        return self.savespace  # C x S x D - z5 in the paper


class TTM(nn.Module):  # Temporal transformer module
    def __init__(self, input, num_submatrices, num_blocks, num_heads, dtype):  # input -> # C x S x D
        super(TTM, self).__init__()
        self.dtype = dtype
        self.avgf = num_submatrices  # average factor (M)
        self.input = input.transpose(0, 2)  # D x S x C
        self.seg = self.input.shape[0] / self.avgf
        if self.input.shape[0] % self.avgf != 0 and int(self.input.shape[0] / self.avgf) == 0:
            print("ERROR 3")

        self.inputc = torch.zeros(self.avgf, self.input.shape[1], self.input.shape[2], dtype=self.dtype).to(device)  # M x S x C

        self.altx = self.inputc.reshape(self.avgf, self.input.shape[1] * self.input.shape[2]).to(device)

        self.inputcf = self.inputc.reshape(self.avgf, self.input.shape[1] * self.input.shape[2])  # M x L1 -> M x (S*C)
        self.inputshape = self.inputcf.shape  # M x L1
        # self.M_size1 = M_size1 # M_size1 = self.eD -> D
        self.M_size1 = self.inputshape[1]  # L1

        self.weight = nn.Parameter(torch.randn(self.M_size1, self.inputshape[1], dtype=self.dtype))  # D x L
        self.bias = torch.zeros(self.inputshape[0], self.M_size1, dtype=self.dtype).to(device)  # ei ∈ R^D -> e ∈ M x D
        self.savespace = torch.zeros(self.inputshape[0], self.M_size1, dtype=self.dtype).to(device)  # M x D

        self.tK = num_blocks  # number of transformer blocks - K in the paper
        self.hA = num_heads  # number of multi-head self-attention units (A is the number of units in a block)
        self.Dh = int(self.M_size1 / self.hA)

        if self.M_size1 % self.hA != 0 and int(self.M_size1 / self.hA) == 0:
            print("ERROR 4")

        self.Wqkv = nn.Parameter(torch.randn((self.tK, 3, self.hA, self.Dh, self.M_size1), dtype=self.dtype))
        self.Wo = nn.Parameter(torch.randn(self.tK, self.M_size1, self.M_size1, dtype=self.dtype)).to(device)
        self.lnorm = nn.LayerNorm(self.M_size1, dtype=self.dtype)  # LayerNorm operation for dimension D
        self.lnormz = nn.LayerNorm(self.M_size1, dtype=self.dtype)  # LayerNorm operation for z
        self.softmax = nn.Softmax()

        self.rsaspace = torch.zeros(self.tK, self.inputshape[0], self.hA, dtype=self.dtype).to(device)  # space for attention score
        self.qkvspace = torch.zeros(self.tK, 3, self.inputshape[0], self.hA, self.Dh, dtype=self.dtype).to(device)  # Q, K, V
        self.imv = torch.zeros(self.tK, self.inputshape[0], self.hA, self.Dh, dtype=self.dtype).to(device)

        self.mlp = Mlp(in_features=self.M_size1, hidden_features=int(self.M_size1 * 4), act_layer=nn.GELU, dtype=self.dtype)  # mlp_ratio=4

    def forward(self, x):
        self.input = x.transpose(0, 2)  # D x S x C

        for i in range(0, self.avgf):  # each i consists self.input.shape[0]/avgf
            for j in range(int(i * self.seg), int((i + 1) * self.seg)):  # int(i*self.seg), int((i+1)*self.seg)
                self.inputc[i, :, :] = self.inputc[i, :, :] + self.input[j, :, :]
            self.inputc[i, :, :] = self.inputc[i, :, :] / self.seg

        self.altx = self.inputc.reshape(self.avgf, self.input.shape[1] * self.input.shape[2])  # M x L -> M x (S*C)
        self.savespace = torch.einsum('lm,im -> il', self.weight, self.altx.clone())

        # - Bias
        for i in range(self.inputshape[0]):  # (i = 1,2,3,…,M)
            for j in range(0, self.M_size1, 2):  # (j = 1,2,3,…,D)
                self.bias[i][j] = math.sin(i / 10000.0 ** (2 * j / self.M_size1))
                self.bias[i][j + 1] = math.cos(i / 10000.0 ** (2 * j / self.M_size1))

        self.savespace = torch.add(self.savespace, self.bias)  # z -> M x D

        for a in range(self.tK):  # blocks(layer)
            self.qkvspace[a] = torch.einsum('xhdm,im -> xihd', self.Wqkv[a], self.lnorm(self.savespace))  # Q, K, V

            # - Attention score
            self.rsaspace[a] = torch.einsum('ihd,ihd -> ih', self.qkvspace[a, 0].clone() / math.sqrt(self.Dh), self.qkvspace[a, 1].clone())

            # - Intermediate vectors
            self.imv[a] = torch.einsum('ih,ihd -> ihd', self.rsaspace[a].clone(), self.qkvspace[a, 2].clone())

            for subj in range(1, self.inputshape[0]):
                self.imv[a, subj] = self.imv[a, subj] + self.imv[a, subj - 1]

            # - NOW SAY HELLO TO NEW Z!
            self.savespace = torch.einsum('nm,im -> in', self.Wo[a], self.imv.clone().reshape(self.tK, self.inputshape[0], self.M_size1)[a]) + self.savespace  # z'

            # - normalized by LN() and passed through a multilayer perceptron (MLP)
            self.savespace = self.lnormz(self.savespace) + self.savespace
            self.savespace = self.mlp(self.savespace)  # new z

        return self.savespace.reshape(self.avgf, self.input.shape[1], self.input.shape[2])


class CNNdecoder(nn.Module):  # EEGformer decoder
    def __init__(self, input, CF_second, dtype):  # input -> # M x S x C
        super(CNNdecoder, self).__init__()
        self.input = input.transpose(0, 1).transpose(1, 2)  # S x C x M
        self.s = self.input.shape[0]
        self.c = self.input.shape[1]
        self.m = self.input.shape[2]
        self.n = CF_second
        self.dtype = dtype
        self.cvd1 = nn.Conv1d(in_channels=self.c, out_channels=1, kernel_size=1, dtype=self.dtype)  # S x M
        self.cvd2 = nn.Conv1d(in_channels=self.s, out_channels=self.n, kernel_size=1, dtype=self.dtype)
        self.cvd3 = nn.Conv1d(in_channels=self.m, out_channels=int(self.m / 2), kernel_size=1, dtype=self.dtype)
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

        self.outshape1 = torch.zeros(self.input_channels, self.ncf, input.shape[0] - 3 * (self.kernel_size - 1)).to(device)
        self.outshape2 = torch.zeros(self.outshape1.shape[0], self.outshape1.shape[1], self.outshape1.shape[2]).to(device)
        self.outshape3 = torch.zeros(self.outshape2.shape[1], self.outshape2.shape[0], self.outshape2.shape[2]).to(device)
        self.outshape4 = torch.zeros(self.avgf, self.outshape3.shape[1], self.outshape3.shape[0]).to(device)

        self.odcm = ODCM(input_channels, self.kernel_size, self.dtype)
        self.rtm = RTM(self.outshape1, self.tK, self.hA, self.dtype)
        self.stm = STM(self.outshape2, self.tK, self.hA, self.dtype)
        self.ttm = TTM(self.outshape3, self.avgf, self.tK, self.hA, self.dtype)
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
        self.lrc = L1_reg_const
        self.w = nn.Parameter(torch.tensor(w))

    def forward(self, x, label):
        x = torch.mean(-torch.log(x @ label) + self.lrc * torch.abs(self.w))
        return x