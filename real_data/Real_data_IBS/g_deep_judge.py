# %% -------------import packages--------------
import torch
from torch import nn
from I_spline import I_U
import numpy as np
#%% -------------------------
def g_D_judge(Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,X_data1,X_data2,X_data3,theta,C,m,nodevec,n_layer,n_node,n_lr,n_epoch):
    Z_train = torch.Tensor(Z_train)
    X_train = torch.Tensor(X_train)
    U_train = torch.Tensor(U_train)
    V_train = torch.Tensor(V_train)
    De1_train = torch.Tensor(De1_train)
    De2_train = torch.Tensor(De2_train)
    De3_train = torch.Tensor(De3_train)
    X_data1 = torch.Tensor(X_data1)
    X_data2 = torch.Tensor(X_data2)
    X_data3 = torch.Tensor(X_data3)
    theta = torch.Tensor(theta)
    C = torch.Tensor(C)
    d = X_train.size()[1]
    # ---------------------------
    class DNNModel(torch.nn.Module):
        def __init__(self):
            super(DNNModel, self).__init__()
            layers = []
            layers.append(nn.Linear(d, n_node))
            layers.append(nn.ReLU())
            for i in range(n_layer):
                layers.append(nn.Linear(n_node, n_node))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(n_node, 2))
            self.model = nn.Sequential(*layers)
        def forward(self, x):
            y_pred = self.model(x)
            return y_pred


    # ----------------------------
    model = DNNModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=n_lr)


    def my_loss(De1,De2,De3,Z,theta,C,m,U,V,nodevec,g_X):
        Iu = torch.Tensor(I_U(m, U.detach().numpy() * (torch.exp(torch.matmul(Z_train, theta[0:4]) + g_X[:,0])).detach().numpy(), nodevec))
        Iv = torch.Tensor(I_U(m, V.detach().numpy() * (torch.exp(torch.matmul(Z_train, theta[0:4]) + g_X[:,0])).detach().numpy(), nodevec))
        Ezg = torch.exp(torch.matmul(Z_train, theta[4:8]) + g_X[:,1])
        loss_fun = - torch.mean(De1 * torch.log(1 - torch.exp(- torch.matmul(Iu,C) * Ezg) + 1e-4) + De2 * torch.log(torch.exp(- torch.matmul(Iu,C) * Ezg) - torch.exp(- torch.matmul(Iv,C) * Ezg) + 1e-4) - De3 * torch.matmul(Iv,C) * Ezg)
        return loss_fun


    # -----------------------------
    for epoch in range(n_epoch):
        pred_g_X = model(X_train)
        loss = my_loss(De1_train,De2_train,De3_train,Z_train,theta,C,m,U_train,V_train,nodevec,pred_g_X)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    # %% -----------------------
    g_train = model(X_train)
    g_value1 = model(X_data1)
    g_value2 = model(X_data2)
    g_value3 = model(X_data3)
    g_train = g_train.detach().numpy()
    g_value1 = g_value1.detach().numpy()
    g_value2 = g_value2.detach().numpy()
    g_value3 = g_value3.detach().numpy()
    return {
        'g_train': g_train,
        'g_value1': g_value1,
        'g_value2': g_value2,
        'g_value3': g_value3
    }
