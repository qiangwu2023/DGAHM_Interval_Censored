
import numpy as np
import scipy.optimize as spo
from I_spline import I_U
from B_spline2 import B_S2


def C_est_median(m, U, V, De1, De2, De3, g_X, nodevec, C0):
    Iu = I_U(m, U * np.exp(g_X[:,0]), nodevec)
    Iv = I_U(m, V * np.exp(g_X[:,0]), nodevec)
    Iuv_2 = I_U(m, (U + V) * np.exp(g_X[:,0]) / 2, nodevec)
    B_U_V_2 = B_S2(m, (U + V) * np.exp(g_X[:,0]) / 2, nodevec)

    def LF(*args):
        a = args[0]
        Ezg = np.exp(g_X[:,1])
        Loss_F1 = - np.mean(De1 * np.log(1 - np.exp(- np.maximum(np.dot(Iu,a) * Ezg, 0)) + 1e-4) + De2 * (np.log(np.dot(B_U_V_2, a) + 1e-4) + g_X[:,0] + g_X[:,1] - np.dot(Iuv_2,a) * Ezg) - De3 * np.dot(Iv,a) * Ezg)
        return Loss_F1
    bnds = [(0, 100)] * (m+3)
    result = spo.minimize(LF, C0, method='SLSQP', bounds=bnds)
    return result['x']





# import numpy as np
# import scipy.optimize as spo
# from I_spline import I_U
# from B_spline2 import B_S2

# def C_est_median(m, U, V, De1, De2, De3, g_X, nodevec, C0, eps=1e-12):
#     # 预计算
#     Ez0 = np.exp(g_X[:, 0])     # exp(z0)
#     Ez1 = np.exp(g_X[:, 1])     # exp(z1)

#     Ue = U * Ez0
#     Ve = V * Ez0
#     Me = (U + V) * Ez0 / 2.0    # 中位时间的缩放输入

#     Iu = I_U(m, Ue, nodevec)
#     Iv = I_U(m, Ve, nodevec)
#     Iuv_2 = I_U(m, Me, nodevec)
#     B_U_V_2 = B_S2(m, Me, nodevec)

#     def LF(a):
#         # 线性组合并确保非负（模型里已用 max(.,0)）
#         xu = np.maximum(Iu @ a, 0.0) * Ez1
#         xv = np.maximum(Iv @ a, 0.0) * Ez1
#         xm = np.maximum(Iuv_2 @ a, 0.0) * Ez1

#         # p1 = 1 - exp(-xu) = -expm1(-xu)
#         p1 = -np.expm1(-xu)
#         p1 = np.clip(p1, eps, 1.0 - eps)

#         # 对 B_U_V_2 @ a 做下限裁剪，避免 log(0)
#         bu = B_U_V_2 @ a
#         bu = np.clip(bu, eps, None)

#         # Loss
#         term1 = De1 * np.log(p1)                    # log(1 - exp(-xu))
#         term2 = De2 * (np.log(bu) + g_X[:, 0] + g_X[:, 1] - xm)  # log(bu) + z0 + z1 - xm
#         term3 = -De3 * xv

#         loss = -np.mean(term1 + term2 + term3)
#         return loss

#     bnds = [(0.0, 100.0)] * (m + 3)
#     res = spo.minimize(LF, C0, method='SLSQP', bounds=bnds)
#     return res.x

