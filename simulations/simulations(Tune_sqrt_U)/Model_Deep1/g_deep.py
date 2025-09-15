# %% -------------import packages--------------
import torch
from torch import nn
from I_spline import I_U
import numpy as np
#%% --------------------------
def g_D(train_data,X_test,X_subject,theta,theta0,C,m,nodevec,n_layer,n_node,n_lr,n_epoch):
    Z_train = torch.Tensor(train_data['Z'])
    X_train = torch.Tensor(train_data['X'])
    U_train = torch.Tensor(train_data['U'])
    V_train = torch.Tensor(train_data['V'])
    De1_train = torch.Tensor(train_data['De1'])
    De2_train = torch.Tensor(train_data['De2'])
    De3_train = torch.Tensor(train_data['De3'])
    g_train_true = torch.Tensor(np.c_[train_data['g1_X'], train_data['g2_X']])
    X_test = torch.Tensor(X_test)
    X_subject = torch.Tensor(X_subject)
    theta = torch.Tensor(theta)
    C = torch.Tensor(C)
    theta0 = torch.Tensor(theta0)
    d = X_train.size()[1]
    # ----------------------------
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
        Iu = torch.Tensor(I_U(m, U.detach().numpy() * (torch.exp(Z * theta[0] + g_X[:,0])).detach().numpy(), nodevec))
        Iv = torch.Tensor(I_U(m, V.detach().numpy() * (torch.exp(Z * theta[0] + g_X[:,0])).detach().numpy(), nodevec))
        Ezg = torch.exp(Z * theta[1] + g_X[:,1])
        loss_fun = - torch.mean(De1 * torch.log(1 - torch.exp(- torch.matmul(Iu,C) * Ezg) + 1e-4) + De2 * torch.log(torch.exp(- torch.matmul(Iu,C) * Ezg) - torch.exp(- torch.matmul(Iv,C) * Ezg) + 1e-4) - De3 * torch.matmul(Iv,C) * Ezg)
        return loss_fun


    # -----------------------------
    for epoch in range(n_epoch):
        pred_g_X = model(X_train) 
        loss = my_loss(De1_train,De2_train,De3_train,Z_train,theta,C,m,U_train,V_train,nodevec,pred_g_X)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    # %% ----------- ------------
    g_train = model(X_train)
    g_test = model(X_test)
    g_subject = model(X_subject)
    g_train = g_train.detach().numpy()
    g_test = g_test.detach().numpy()
    g_subject = g_subject.detach().numpy()
    return {
        'g_train': g_train,
        'g_test': g_test,
        'g_subject': g_subject
    }
