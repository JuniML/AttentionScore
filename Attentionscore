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



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# Initialize model
model_feature_encoder_ecfp4 = EncoderLayer(vector_size, n_heads, n_layers)
model_feature_encoder_ecfp4 = EncoderLayer.to(device)

for batch_idx, data in enumerate(loader_ecfp4, 0):
    inputs, targets = data
    
    inputs = inputs.to(device)
    targets = targets.to(device)
    # Forward pass
    X1, X2, X3, X5 = model_feature_encoder_ecfp4(inputs.float())
    
    # Compute loss
    loss = criterion_mse(inputs.float(), X5)
    
    # Backward pass and optimize
    loss.backward()
    model_optimizer_feature_encoder.step()
    
    running_loss += loss.item()
    
    # Average loss for this epoch
    epoch_loss = running_loss / len_train
    training_losses_ecfp4.append(epoch_loss)
    
    print(f'Epoch [{epoch+1}/{epo_num}], Loss: {epoch_loss:.6f}')


# Set model to evaluation mode
model_feature_encoder_ecfp4.eval()

# Initialize lists to store the latent features from all batches
XB1_list = []
XB2_list = []
XB3_list = []
XB5_list = []

with torch.no_grad():  # Disable gradient computation for evaluation
    for inputs, _ in loader_ecfp4:  # Iterate over batches in train_loader
        inputs = inputs.to(device)  # Ensure inputs are on CPU
        XB1, XB2, XB3, XB5 = model_feature_encoder_ecfp4(inputs.float())

        # Detach and move the tensors to CPU, if not already
        XB1 = XB1.detach().cpu()
        XB2 = XB2.detach().cpu()
        XB3 = XB3.detach().cpu()
        XB5 = XB5.detach().cpu()

        # append the tensors to the lists
        XB1_list.append(XB1)
        XB2_list.append(XB2)
        XB3_list.append(XB3)
        XB5_list.append(XB5)

# Concatenate all batches into a single tensor for each latent feature
XB1_all = torch.cat(XB1_list, dim=0)
XB2_all = torch.cat(XB2_list, dim=0)
XB3_all = torch.cat(XB3_list, dim=0)
XB5_all = torch.cat(XB5_list, dim=0)


# Initialize model
model_feature_encoder_plec = feature_encoder(vector_size, n_heads, n_layers)
model_feature_encoder_plec = model_feature_encoder_plec.to(device)

# Loss function
criterion_mse = nn.MSELoss()

# List to store training losses
training_losses_plec = []

# Optimizer
model_optimizer_feature_encoder = optim.Adam(model_feature_encoder_plec.parameters(), lr=learning_rate)

len_train = len(loader_plec)

# Training loop
for epoch in range(epo_num):
    model_feature_encoder_plec.train()
    running_loss = 0.0
    for batch_idx, data in enumerate(loader_plec, 0):
        inputs, targets = data
        
        inputs = inputs.to(device)
        targets = targets.to(device)

        model_optimizer_feature_encoder.zero_grad()

        # Forward pass
        X1, X2, X3, X5 = model_feature_encoder_plec(inputs.float())
        
        # Compute loss
        loss = criterion_mse(inputs.float(), X5)
        
        # Backward pass and optimize
        loss.backward()
        model_optimizer_feature_encoder.step()
        
        running_loss += loss.item()
    
    # Average loss for this epoch
    epoch_loss = running_loss / len_train
    training_losses_plec.append(epoch_loss)
    
    print(f'Epoch [{epoch+1}/{epo_num}], Loss: {epoch_loss:.6f}')

import torch

# Set model to evaluation mode
model_feature_encoder_plec.eval()

# Initialize lists to store the latent features from all batches
XC1_list = []
XC2_list = []
XC3_list = []
XC5_list = []

with torch.no_grad():  # Disable gradient computation for evaluation
    for inputs, _ in loader_plec:  # Iterate over batches in train_loader
        inputs = inputs.to(device)  # Ensure inputs are on CPU
        XC1, XC2, XC3, XC5 = model_feature_encoder_plec(inputs.float())

        # Detach and move the tensors to CPU, if not already
        XC1 = XC1.detach().cpu()
        XC2 = XC2.detach().cpu()
        XC3 = XC3.detach().cpu()
        XC5 = XC5.detach().cpu()

        # append the tensors to the lists
        XC1_list.append(XC1)
        XC2_list.append(XC2)
        XC3_list.append(XC3)
        XC5_list.append(XC5)

# Concatenate all batches into a single tensor for each latent feature
XC1_all = torch.cat(XC1_list, dim=0)
XC2_all = torch.cat(XC2_list, dim=0)
XC3_all = torch.cat(XC3_list, dim=0)
XC5_all = torch.cat(XC5_list, dim=0)

XDC = torch.cat((XC5_all, XB5_all), 1)
# multi scale features fusion
# X1 = torch.cat((XB3_all, XC1_all), 1)
# X2 = torch.cat((XB2_all, XC2_all), 1)
# X3 = torch.cat((XB1_all, XC3_all), 1)

# single scale features fusion
X1 = torch.cat((XB1_all, XC1_all), 1)
X2 = torch.cat((XB2_all, XC2_all), 1)
X3 = torch.cat((XB3_all, XC3_all), 1)

X1_dataset = TensorDataset(X1, labels)
X1_loader = DataLoader(dataset=X1_dataset, batch_size=32, shuffle=True)

X2_dataset = TensorDataset(X2, labels)
X2_loader = DataLoader(dataset=X2_dataset, batch_size=32, shuffle=True)

X3_dataset = TensorDataset(X3, labels)
X3_loader = DataLoader(dataset=X3_dataset, batch_size=32, shuffle=True)


# Initialize lists to store the latent features from all batches
FD1_list = []
vector_size = X1.shape[1]
model_feature_encoder2 = feature_encoder2(vector_size)
model_feature_encoder2 = model_feature_encoder2.to(device)
for inputs, _ in X1_loader:  # Iterate over batches in train_loader
    inputs = inputs.to(device)  # Ensure inputs are on CPU
    FD1 = model_feature_encoder2(inputs.float())
    # Detach and move the tensors to CPU, if not already
    FD1 = FD1.detach().cpu()
    # append the tensors to the lists
    FD1_list.append(FD1)

# Concatenate all batches into a single tensor for each latent feature
FD1_all = torch.cat(FD1_list, dim=0)

# # Save the tensors to disk
# torch.save(FD1_all, '/home/juni/working/mettl3/FD1_all.pt')
# print("Tensors saved successfully.")

# Initialize lists to store the latent features from all batches
FD2_list = []
vector_size = X2.shape[1]
model_feature_encoder2 = feature_encoder2(vector_size)
model_feature_encoder2 = model_feature_encoder2.to(device)
for inputs, _ in X2_loader:  # Iterate over batches in train_loader
    inputs = inputs.to(device)  # Ensure inputs are on CPU
    FD2 = model_feature_encoder2(inputs.float())
    # Detach and move the tensors to CPU, if not already
    FD2 = FD2.detach().cpu()
    # append the tensors to the lists
    FD2_list.append(FD2)

# Concatenate all batches into a single tensor for each latent feature
FD2_all = torch.cat(FD2_list, dim=0)

# # Save the tensors to disk
# torch.save(FD2_all, '/home/juni/working/mettl3/FD2_all.pt')
# print("Tensors saved successfully.")

# Initialize lists to store the latent features from all batches
FD3_list = []
vector_size = X3.shape[1]
model_feature_encoder2 = feature_encoder2(vector_size)
model_feature_encoder2 = model_feature_encoder2.to(device)
for inputs, _ in X3_loader:  # Iterate over batches in train_loader
    inputs = inputs.to(device)  # Ensure inputs are on CPU
    FD3 = model_feature_encoder2(inputs.float())
    # Detach and move the tensors to CPU, if not already
    FD3 = FD3.detach().cpu()
    # append the tensors to the lists
    FD3_list.append(FD3)

# Concatenate all batches into a single tensor for each latent feature
FD3_all = torch.cat(FD3_list, dim=0)

# # Save the tensors to disk
# torch.save(FD3_all, '/home/juni/working/mettl3/FD3_all.pt')
# print("Tensors saved successfully.")

# XC = torch.cat((FD1_all, FD2_all, FD3_all, XB3_all, XC3_all), 1)
XC = torch.cat((FD3_all, XB3_all, XC3_all), 1)
print(XC.shape)

import torch
import numpy as np
from sklearn.model_selection import train_test_split
# Convert tensors to numpy arrays for splitting
XC = XC.detach().cpu()
labels = labels.detach().cpu()

XC_np = XC.numpy()
y_np = labels.numpy()

# Split data into train and test sets (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(XC_np, y_np, test_size=0.2, random_state=42)
