# ----------import packages------------
import numpy as np
from C_estimation import C_est
from g_linear_judge import g_L_judge


def Est_linear_judge(Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,theta_initial,nodevec,m,C):
    C_index = 0
    for loop in range(100):
        print('linear_iteration_judge time=', loop)
        g_X = g_L_judge(Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,C,m,nodevec)
        g_train = g_X['g_train']
        g_parameter = g_X['linear_g']
        theta = g_X['theta']
        C = C_est(m, U_train, V_train, De1_train, De2_train, De3_train, Z_train, theta, g_train, nodevec)
        if (np.max(np.abs(theta - theta_initial)) <= 0.01 and loop > 3):
            C_index = 1
            break
        theta_initial = theta
    
    return {'g_parameter': g_parameter,
        'g_train': g_train,
        'C': C,
        'theta': theta,
        'C_index': C_index
    }
