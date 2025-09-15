# %% -------------import packages--------------
import numpy as np
import scipy.optimize as spo
from I_spline import I_U
from B_spline3 import B_S

def g_A(Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,X_sort1,X_sort0,theta,C,m,nodevec,m0,nodevec0):
    B_0 = B_S(m0, X_train[:,0], nodevec0)
    B_1 = B_S(m0, X_train[:,1], nodevec0)
    B_2 = B_S(m0, X_train[:,2], nodevec0)
    B_3 = B_S(m0, X_train[:,3], nodevec0)
    B_0_test = B_S(m0, X_test[:,0], nodevec0)
    B_1_test = B_S(m0, X_test[:,1], nodevec0)
    B_2_test = B_S(m0, X_test[:,2], nodevec0)
    B_3_test = B_S(m0, X_test[:,3], nodevec0)
    B_0_subject = B_S(m0, X_subject[:,0], nodevec0)
    B_1_subject = B_S(m0, X_subject[:,1], nodevec0)
    B_2_subject = B_S(m0, X_subject[:,2], nodevec0)
    B_3_subject = B_S(m0, X_subject[:,3], nodevec0)
    d = X_train.shape[1]
    def GA(*args):
        b = args[0]
        g1_X = np.dot(B_0, b[0:(m0+4)]) + np.dot(B_1, b[(m0+4):(2*(m0+4))]) + np.dot(B_2, b[(2*(m0+4)):(3*(m0+4))]) + np.dot(B_3, b[(3*(m0+4)):(4*(m0+4))]) + b[4*(m0+4)]*np.ones(X_train.shape[0])

        g2_X = np.dot(B_0, b[(4*(m0+4)+1):(5*(m0+4)+1)]) + np.dot(B_1, b[(5*(m0+4)+1):(6*(m0+4)+1)]) + np.dot(B_2, b[(6*(m0+4)+1):(7*(m0+4)+1)]) + np.dot(B_3, b[(7*(m0+4)+1):(8*(m0+4)+1)]) + b[8*(m0+4)+1]*np.ones(X_train.shape[0])

        Iu = I_U(m, np.clip(U_train * np.exp(Z_train * theta[0] + g1_X), 0, 90), nodevec)
        Iv = I_U(m, np.clip(V_train * np.exp(Z_train * theta[0] + g1_X), 0, 90), nodevec)
        Ezg = np.exp(Z_train * theta[1] + g2_X)
        loss_fun = - np.mean(De1 * np.log(1 - np.exp(- np.dot(Iu, C) * Ezg) + 1e-4) + De2 * np.log(np.exp(- np.dot(Iu, C) * Ezg) - np.exp(- np.dot(Iv, C) * Ezg) + 1e-4) - De3 * np.dot(Iv, C) * Ezg)
        return loss_fun
    g_additive_para = spo.minimize(GA,np.zeros(8*(m0+4)+2),method='SLSQP')['x']
    
    g1_train = np.dot(B_0, g_additive_para[0:(m0+4)]) + np.dot(B_1, g_additive_para[(m0+4):(2*(m0+4))]) + np.dot(B_2, g_additive_para[(2*(m0+4)):(3*(m0+4))]) + np.dot(B_3, g_additive_para[(3*(m0+4)):(4*(m0+4))]) + g_additive_para[4*(m0+4)]*np.ones(X_train.shape[0])  
    g2_train = np.dot(B_0, g_additive_para[(4*(m0+4)+1):(5*(m0+4)+1)]) + np.dot(B_1, g_additive_para[(5*(m0+4)+1):(6*(m0+4)+1)]) + np.dot(B_2, g_additive_para[(6*(m0+4)+1):(7*(m0+4)+1)]) + np.dot(B_3, g_additive_para[(7*(m0+4)+1):(8*(m0+4)+1)]) + g_additive_para[8*(m0+4)+1]*np.ones(X_train.shape[0])
    g_train = np.c_[g1_train, g2_train]


    g1_test = np.dot(B_0_test, g_additive_para[0:(m0+4)]) + np.dot(B_1_test, g_additive_para[(m0+4):(2*(m0+4))]) + np.dot(B_2_test, g_additive_para[(2*(m0+4)):(3*(m0+4))]) + np.dot(B_3_test, g_additive_para[(3*(m0+4)):(4*(m0+4))]) + g_additive_para[4*(m0+4)]*np.ones(X_test.shape[0])
    g2_test = np.dot(B_0_test, g_additive_para[(4*(m0+4)+1):(5*(m0+4)+1)]) + np.dot(B_1_test, g_additive_para[(5*(m0+4)+1):(6*(m0+4)+1)]) + np.dot(B_2_test, g_additive_para[(6*(m0+4)+1):(7*(m0+4)+1)]) + np.dot(B_3_test, g_additive_para[(7*(m0+4)+1):(8*(m0+4)+1)]) + g_additive_para[8*(m0+4)+1]*np.ones(X_test.shape[0])
    g_test = np.c_[g1_test, g2_test]

    g1_subject = np.dot(B_0_subject, g_additive_para[0:(m0+4)]) + np.dot(B_1_subject, g_additive_para[(m0+4):(2*(m0+4))]) + np.dot(B_2_subject, g_additive_para[(2*(m0+4)):(3*(m0+4))]) + np.dot(B_3_subject, g_additive_para[(3*(m0+4)):(4*(m0+4))]) + g_additive_para[4*(m0+4)]*np.ones(X_subject.shape[0])
    g2_subject = np.dot(B_0_subject, g_additive_para[(4*(m0+4)+1):(5*(m0+4)+1)]) + np.dot(B_1_subject, g_additive_para[(5*(m0+4)+1):(6*(m0+4)+1)]) + np.dot(B_2_subject, g_additive_para[(6*(m0+4)+1):(7*(m0+4)+1)]) + np.dot(B_3_subject, g_additive_para[(7*(m0+4)+1):(8*(m0+4)+1)]) + g_additive_para[8*(m0+4)+1]*np.ones(X_subject.shape[0])
    g_subject = np.c_[g1_subject, g2_subject]

    return {'additive_g': g_additive_para,
        'g_train': g_train,
        'g_test': g_test,
        'g_subject': g_subject
    }
