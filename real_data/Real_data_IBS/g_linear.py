# # %% -------------import packages--------------
# import numpy as np
# import scipy.optimize as spo
# from I_spline import I_U

# def g_L(W_train,U_train,V_train,De1_train,De2_train,De3_train,W_sort1,W_sort0,C,m,nodevec):
#     d = W_train.shape[1]
#     def GL(*args):
#         b = args[0]
#         g1_X = np.dot(W_train,b[0:d]) + b[d]*np.ones(W_train.shape[0])
#         g2_X = np.dot(W_train,b[(d+1):(2*d+1)]) + b[2*d+1]*np.ones(W_train.shape[0])
#         Iu = I_U(m, U_train * np.exp(g1_X), nodevec)
#         Iv = I_U(m, V_train * np.exp(g1_X), nodevec)
#         Ezg = np.exp(g2_X)
#         loss_fun = - np.mean(De1_train * np.log(1 - np.exp(- np.dot(Iu, C) * Ezg) + 1e-4) + De2_train * np.log(np.exp(- np.dot(Iu, C) * Ezg) - np.exp(- np.dot(Iv, C) * Ezg) + 1e-4) - De3_train * np.dot(Iv, C) * Ezg)
#         return loss_fun
#     g_linear_para = spo.minimize(GL,np.zeros(2*d+2),method='SLSQP')['x']
    
#     g1_train = np.dot(W_train, g_linear_para[0:d]) + g_linear_para[d]*np.ones(W_train.shape[0])
#     g2_train = np.dot(W_train, g_linear_para[(d+1):(2*d+1)]) + g_linear_para[2*d+1]*np.ones(W_train.shape[0])
#     g_train = np.c_[g1_train, g2_train]
    
#     g1_test1 = np.dot(W_sort1, g_linear_para[0:d]) + g_linear_para[d]*np.ones(W_sort1.shape[0])
#     g2_test1 = np.dot(W_sort1, g_linear_para[(d+1):(2*d+1)]) + g_linear_para[2*d+1]*np.ones(W_sort1.shape[0])
#     g_test1 = np.c_[g1_test1, g2_test1]

#     g1_test0 = np.dot(W_sort0, g_linear_para[0:d]) + g_linear_para[d]*np.ones(W_sort0.shape[0])
#     g2_test0 = np.dot(W_sort0, g_linear_para[(d+1):(2*d+1)]) + g_linear_para[2*d+1]*np.ones(W_sort0.shape[0])
#     g_test0 = np.c_[g1_test0, g2_test0]


#     return {'linear_g': g_linear_para,
#             'g_train': g_train,
#             'g_test1': g_test1,
#             'g_test0': g_test0
#     }



import numpy as np
import scipy.optimize as spo
from I_spline import I_U

def g_L(W_train, U_train, V_train, De1_train, De2_train, De3_train, W_test, W_sort1, W_sort0, C, m, nodevec):
    d = W_train.shape[1]
    eps = 1e-12
    exp_cap = 50.0  # 限制指数，避免 exp 溢出，exp(50)≈3.0e21

    def GL(*args):
        b = args[0]

        # 线性项
        g1_X = W_train @ b[0:d] + b[d]
        g2_X = W_train @ b[(d+1):(2*d+1)] + b[2*d+1]

        # 限制指数范围，防止溢出
        eg1 = np.exp(np.clip(g1_X, -exp_cap, exp_cap))
        Ezg = np.exp(np.clip(g2_X, -exp_cap, exp_cap))

        # I-spline 基函数线性组合
        Iu = I_U(m, U_train * eg1, nodevec)  # shape: (n, K)
        Iv = I_U(m, V_train * eg1, nodevec)  # shape: (n, K)
        IuC = Iu @ C                           # shape: (n,)
        IvC = Iv @ C

        # 共同项
        a = np.clip(IuC * Ezg, 0.0, np.inf)
        b_ = np.clip(IvC * Ezg, 0.0, np.inf)

        # 稳定计算：1 - exp(-a) 用 -expm1(-a)
        p1 = -np.expm1(-a)                    # = 1 - exp(-a), in (0,1)
        # exp(-a) - exp(-b) = exp(-b) * (exp(-(a-b)) - 1) = -exp(-b) * expm1(-(a-b))
        # 直接稳定算：先算 e^{-a}, e^{-b}
        ea = np.exp(-np.clip(a, 0.0, exp_cap))
        eb = np.exp(-np.clip(b_, 0.0, exp_cap))
        p2 = ea - eb

        # 夹紧，避免 log(0)
        p1 = np.clip(p1, eps, 1.0)            # 上界也夹到1，避免微小>1
        # p2 需要为正（理论上 a<b 才正），数值误差下可能<=0
        p2 = np.clip(p2, eps, np.inf)

        # 第三项是线性项
        term1 = De1_train * np.log(p1)
        term2 = De2_train * np.log(p2)
        term3 = -De3_train * b_

        loss_fun = -np.mean(term1 + term2 + term3)
        return loss_fun

    res = spo.minimize(GL, np.zeros(2*d + 2), method='SLSQP')
    g_linear_para = res['x']

    # 训练与测试的 g 值
    g1_train = W_train @ g_linear_para[0:d] + g_linear_para[d]
    g2_train = W_train @ g_linear_para[(d+1):(2*d+1)] + g_linear_para[2*d+1]
    g_train = np.c_[g1_train, g2_train]

    g1_test1 = W_sort1 @ g_linear_para[0:d] + g_linear_para[d]
    g2_test1 = W_sort1 @ g_linear_para[(d+1):(2*d+1)] + g_linear_para[2*d+1]
    g_test1 = np.c_[g1_test1, g2_test1]

    g1_test0 = W_sort0 @ g_linear_para[0:d] + g_linear_para[d]
    g2_test0 = W_sort0 @ g_linear_para[(d+1):(2*d+1)] + g_linear_para[2*d+1]
    g_test0 = np.c_[g1_test0, g2_test0]

    g1_test = W_test @ g_linear_para[0:d] + g_linear_para[d]
    g2_test = W_test @ g_linear_para[(d+1):(2*d+1)] + g_linear_para[2*d+1]
    g_test = np.c_[g1_test, g2_test]

    return {
        'linear_g': g_linear_para,
        'g_train': g_train,
        'g_test1': g_test1,
        'g_test0': g_test0,
        'g_test': g_test
    }

