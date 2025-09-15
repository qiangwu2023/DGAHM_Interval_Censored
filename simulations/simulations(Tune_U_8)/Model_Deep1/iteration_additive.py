# ----------import packages------------
import numpy as np
from C_estimation import C_est
from g_additive import g_A

def Est_additive(train_data,X_test,X_subject,theta_initial,nodevec,m,C,m0,nodevec0):
    Z_train = train_data['Z']
    U_train = train_data['U']
    V_train = train_data['V']
    De1_train = train_data['De1']
    De2_train = train_data['De2']
    De3_train = train_data['De3']
    C_index = 0
    for loop in range(10):
        print('additive_iteration time=', loop)
        g_X = g_A(train_data,X_test,X_subject,C,m,nodevec,m0,nodevec0)
        g_train = g_X['g_train']
        theta = g_X['theta']
        print('theta=', theta)
        C = C_est(m, U_train, V_train, De1_train, De2_train, De3_train, Z_train, theta, g_train, nodevec)
        if (np.abs(theta[0] - theta_initial[0]) <= 0.01 and np.abs(theta[1] - theta_initial[1]) <= 0.01 and loop > 3):
            C_index = 1
            break
        theta_initial = theta
    
    return {
        'g_train': g_train,
        'g_test': g_X['g_test'],
        'g_subject': g_X['g_subject'],
        'C': C,
        'theta': theta,
        'C_index': C_index
    }
