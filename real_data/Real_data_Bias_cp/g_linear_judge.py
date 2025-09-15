# %% -------------import packages--------------
import numpy as np
import scipy.optimize as spo
from I_spline import I_U

def g_L_judge(Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,C,m,nodevec):
    c = Z_train.shape[1]
    d = X_train.shape[1]
    def GL(*args):
        b = args[0]
        g1_X = np.dot(X_train,b[0:d]) + b[d]*np.ones(X_train.shape[0])
        g2_X = np.dot(X_train,b[(d+1):(2*d+1)]) + b[2*d+1]*np.ones(X_train.shape[0])
        Iu = I_U(m, U_train * np.exp(np.dot(Z_train, b[(2*d+2):(2*d+2+c)]) + g1_X), nodevec)
        Iv = I_U(m, V_train * np.exp(np.dot(Z_train, b[(2*d+2):(2*d+2+c)]) + g1_X), nodevec)
        Ezg = np.exp(np.dot(Z_train, b[(2*d+2+c):(2*d+2+2*c)]) + g2_X)
        loss_fun = - np.mean(De1_train * np.log(1 - np.exp(- np.dot(Iu, C) * Ezg) + 1e-4) + De2_train * np.log(np.exp(- np.dot(Iu, C) * Ezg) - np.exp(- np.dot(Iv, C) * Ezg) + 1e-4) - De3_train * np.dot(Iv, C) * Ezg)
        return loss_fun
    g_linear_para = spo.minimize(GL,np.zeros(2*d+2+2*c),method='SLSQP')['x']
    
    g1_train = np.dot(X_train, g_linear_para[0:d]) + g_linear_para[d]*np.ones(X_train.shape[0])
    g2_train = np.dot(X_train, g_linear_para[(d+1):(2*d+1)]) + g_linear_para[2*d+1]*np.ones(X_train.shape[0])
    g_train = np.c_[g1_train, g2_train]

    theta = g_linear_para[(2*d+2):(2*d+2+2*c)]
    return {'linear_g': g_linear_para[(d+1):(2*d+1)],
            'theta': theta,
            'g_train': g_train
    }
