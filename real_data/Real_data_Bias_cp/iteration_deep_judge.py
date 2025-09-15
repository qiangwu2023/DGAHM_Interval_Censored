import numpy as np
from beta_estimate import beta_est
from gamma_estimate import gamma_est
from C_estimation import C_est
from g_deep_judge import g_D_judge


def Est_deep_judge(Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,X_data1,X_data2,X_data3,theta_initial,n_layer,n_node,n_lr,n_epoch,nodevec,m,C):
    C_index = 0
    d = Z_train.shape[1]
    for loop in range(100):
        print('deep_iteration_judge time=', loop)
        g_X = g_D_judge(Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,X_data1,X_data2,X_data3,theta_initial,C,m,nodevec,n_layer,n_node,n_lr,n_epoch)
        g_train = g_X['g_train']
        C = C_est(m, U_train, V_train, De1_train, De2_train, De3_train, Z_train, theta_initial, g_train, nodevec)
        print('C=', C)
        beta = beta_est(theta_initial[d:(2*d)], De1_train, De2_train, De3_train, Z_train, U_train, V_train, C, m, g_train, nodevec)
        gamma = gamma_est(beta, De1_train, De2_train, De3_train, Z_train, U_train, V_train, C, m, g_train, nodevec)
        theta = np.array([beta, gamma]).reshape(2*d,)
        print('theta=', theta)
        print('max_error=', np.max(np.abs(theta - theta_initial)))
        if (np.max(np.abs(theta - theta_initial)) <= 0.01 and loop > 3):
            C_index = 1
            break
        theta_initial = theta
    
    return {
        'g_train': g_train,
        'g_value1': g_X['g_value1'],
        'g_value2': g_X['g_value2'],
        'g_value3': g_X['g_value3'],
        'C': C,
        'theta': theta,
        'C_index': C_index
    }

