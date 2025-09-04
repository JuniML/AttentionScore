from __future__ import annotations

import math
import torch
from torch import nn
import numpy as np

drop_out_rating: float = 0.001

import torch
from torch import nn
class MultiHeadAttention(torch.nn.Module):
    def __init__(self,input_dim,n_heads,ouput_dim=None):
        
        super(MultiHeadAttention, self).__init__()
        self.d_k=self.d_v=input_dim//n_heads
        self.n_heads = n_heads
        if ouput_dim==None:
            self.ouput_dim=input_dim
        else:
            self.ouput_dim=ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)
    def forward(self,X):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q=self.W_Q(X).view( -1, self.n_heads, self.d_k).transpose(0,1)
        K=self.W_K(X).view( -1, self.n_heads, self.d_k).transpose(0,1)
        V=self.W_V(X).view( -1, self.n_heads, self.d_v).transpose(0,1)
        
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        # context: [len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output


# In[107]:


class EncoderLayer(torch.nn.Module):
    def __init__(self,input_dim,n_heads):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(input_dim,n_heads)
        self.AN1=torch.nn.LayerNorm(input_dim)
        
        self.l1=torch.nn.Linear(input_dim, input_dim)
        self.AN2=torch.nn.LayerNorm(input_dim)
    def forward (self,X):
        
        output=self.attn(X)
        X=self.AN1(output+X)
        
        output=self.l1(X)
        X=self.AN2(output+X)
        
        return X


# In[108]:


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# In[109]:




# In[112]:

class feature_encoder(torch.nn.Module):  # twin network
    def __init__(self, vector_size,n_heads,n_layers):
        super(feature_encoder, self).__init__()

        self.layers = torch.nn.ModuleList([EncoderLayer(vector_size, n_heads) for _ in range(n_layers)])
        self.AN = torch.nn.LayerNorm(vector_size)

        self.l1 = torch.nn.Linear(vector_size, vector_size // 2)
        self.bn1 = torch.nn.BatchNorm1d(vector_size // 2)

        self.l2 = torch.nn.Linear(vector_size // 2, vector_size // 4)

        self.l3 = torch.nn.Linear(vector_size // 4, vector_size//2)
        self.bn3 = torch.nn.BatchNorm1d(vector_size // 2)

        self.l4 = torch.nn.Linear(vector_size // 2, vector_size )


        self.dr = torch.nn.Dropout(drop_out_rating)

        self.ac = gelu

    def forward(self, X):

        for layer in self.layers:
            X = layer(X)
        X1=self.AN(X)
        X2 = self.dr(self.bn1(self.ac(self.l1(X1))))
        X3 = self.l2(X2)

        X4 = self.dr(self.bn3(self.ac(self.l3(self.ac(X3)))))
        X5 = self.l4(X4)

        return X1,X2,X3,X5
class feature_encoder2(torch.nn.Module):  # twin network
    def __init__(self, vector_size):
        super(feature_encoder2, self).__init__()

        self.l1 = torch.nn.Linear(vector_size, vector_size // 2)
        self.bn1 = torch.nn.BatchNorm1d(vector_size // 2)

        self.l2 = torch.nn.Linear(vector_size // 2, vector_size // 4)
        self.bn2 = torch.nn.BatchNorm1d(vector_size // 4)

        self.dr = torch.nn.Dropout(drop_out_rating)

        self.ac = gelu

    def forward(self, X):

        X = self.dr(self.bn1(self.ac(self.l1(X))))

        X = self.dr(self.bn2(self.ac(self.l2(X))))

        return X
class Model(torch.nn.Module):
    def __init__(self,input_dim_A,input_dim_B, n_heads,n_layers,event_num):
        super(Model, self).__init__()

        #self.input_dim = input_dim
        self.input_dim_A = input_dim_A
        self.input_dim_B = input_dim_B
        self.drugEncoder_input_dim_A=self.input_dim_A
        self.drugEncoder_input_dim_B=self.input_dim_B
        self.drugEncoderA=feature_encoder(self.drugEncoder_input_dim_A,n_heads,n_layers)
        self.drugEncoderB = feature_encoder(self.drugEncoder_input_dim_B, n_heads, n_layers)

        self.feaEncoder1_3_input_dim=self.drugEncoder_input_dim_A+self.drugEncoder_input_dim_B//4
        self.feaEncoder3_1_input_dim=self.drugEncoder_input_dim_B+self.drugEncoder_input_dim_A//4
        self.feaEncoder2_input_dim = self.drugEncoder_input_dim_A//2 + self.drugEncoder_input_dim_B// 2

        self.feaEncoder1 = feature_encoder2(self.feaEncoder1_3_input_dim)
        self.feaEncoder2 = feature_encoder2(self.feaEncoder2_input_dim)
        self.feaEncoder3 = feature_encoder2(self.feaEncoder3_1_input_dim)

        self.feaEncoder1_3_output_dim = self.feaEncoder1_3_input_dim//4
        self.feaEncoder3_1_output_dim = self.feaEncoder3_1_input_dim//4
        self.feaEncoder2_output_dim = self.feaEncoder2_input_dim//4

        self.feaFui_input_dim = self.feaEncoder1_3_output_dim+self.feaEncoder3_1_output_dim+self.feaEncoder2_output_dim+self.drugEncoder_input_dim_A//4+self.drugEncoder_input_dim_B//4

        #self.feaFui = feature_encoder(self.feaFui_input_dim, n_heads, n_layers)
        self.feaFui = feature_encoder2(self.feaFui_input_dim)
        self.linear_input_dim = self.feaFui_input_dim//4+self.feaFui_input_dim

        self.l1=torch.nn.Linear(self.linear_input_dim,(self.linear_input_dim)//2)
        self.bn1=torch.nn.BatchNorm1d((self.linear_input_dim)//2)

        self.l2 = torch.nn.Linear((self.linear_input_dim)//2, 1)
        
        self.ac=gelu

        self.dr = torch.nn.Dropout(drop_out_rating)
        self.sigmoid = torch.nn.Sigmoid()  # Sigmoid for probabilities
        
    def forward(self, XA, XB):
        # XA = X[:, 0:self.input_dim//2]
        # XB = X[:, self.input_dim//2:]

        XA1,XA2,XA3,XAC=self.drugEncoderA(XA)
        XB1, XB2, XB3 ,XBC= self.drugEncoderB(XB)

        XDC = torch.cat((XAC, XBC), 1)

        X1 = torch.cat((XA1,XB3), 1)
        X2 = torch.cat((XA2, XB2), 1)
        X3 = torch.cat((XA3, XB1), 1)

        X1=self.feaEncoder1(X1)
        X2 = self.feaEncoder2(X2)
        X3 = self.feaEncoder3(X3)

        XC = torch.cat((X1, X2, X3,XA3,XB3), 1)
        #_,_,XC,_=self.feaFui(XC)
        XC=self.feaFui(XC)

        X = torch.cat((XA3,XB3,X1, X2, X3,XC), 1)

        X=self.dr(self.bn1(self.ac(self.l1(X))))


        X=self.l2(X)
        X= self.sigmoid(X)
        return X,XC,XDC, XAC, XBC