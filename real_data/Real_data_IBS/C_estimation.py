# import numpy as np
# import scipy.optimize as spo
# from I_spline import I_U


# def C_est(m, U, V, De1, De2, De3, Z, theta, g_X, nodevec):
#     c = Z.shape[1]
#     Iu = I_U(m, U * np.exp(np.dot(Z, theta[0:c]) + g_X[:,0]), nodevec)
#     Iv = I_U(m, V * np.exp(np.dot(Z, theta[0:c]) + g_X[:,0]), nodevec)
#     def LF(*args):
#         a = args[0]
#         Ezg = np.exp(np.dot(Z, theta[c:(2*c)]) + g_X[:,1])
#         Loss_F1 = - np.mean(De1 * np.log(1 - np.exp(- np.dot(Iu,a) * Ezg) + 1e-4) + De2 * np.log(np.exp(- np.dot(Iu,a) * Ezg) - np.exp(- np.dot(Iv,a) * Ezg) + 1e-4) - De3 * np.dot(Iv,a) * Ezg)
#         return Loss_F1
#     bnds = [(0, 100)] * (m+3)
#     result = spo.minimize(LF, np.ones(m+3),method='SLSQP',bounds=bnds)
#     return result['x']

import numpy as np
import scipy.optimize as spo
from I_spline import I_U

def C_est(m, U, V, De1, De2, De3, Z, theta, g_X, nodevec, eps=1e-12):
    c = Z.shape[1]

    # 预计算缩放
    scale_u = np.exp(Z @ theta[0:c] + g_X[:, 0])
    scale_v = scale_u  # 同一表达式
    Iu = I_U(m, U * scale_u, nodevec)
    Iv = I_U(m, V * scale_v, nodevec)

    Ezg = np.exp(Z @ theta[c:(2*c)] + g_X[:, 1])

    def LF(a):
        # 非负强制（你的原式中无 maximum，但概率项需要非负强度）
        xu = np.maximum(Iu @ a, 0.0) * Ezg
        xv = np.maximum(Iv @ a, 0.0) * Ezg

        # p1 = 1 - exp(-xu) = -expm1(-xu) ∈ (0,1)
        p1 = -np.expm1(-xu)
        p1 = np.clip(p1, eps, 1.0 - eps)

        # p2 = exp(-xu) - exp(-xv)
        # 稳定写法：p2 = exp(-xu) * (1 - exp(-(xv - xu))) = exp(-xu) * (-expm1(-(xv - xu)))
        delta = xv - xu
        p2 = np.exp(-xu) * (-np.expm1(-np.maximum(delta, 0.0)))  # 若 xv < xu，理论上 p2 可能为负
        # 若模型或数据可能造成 p2<=0，做裁剪保证 log 可用，同时提示：
        p2 = np.clip(p2, eps, 1.0 - eps)

        # 目标函数
        loss = -np.mean(
            De1 * np.log(p1) +
            De2 * np.log(p2) -
            De3 * xv
        )
        return loss

    bnds = [(0.0, 100.0)] * (m + 3)
    res = spo.minimize(LF, 0.1*np.ones(m + 3), method='SLSQP', bounds=bnds)
    return res.x
