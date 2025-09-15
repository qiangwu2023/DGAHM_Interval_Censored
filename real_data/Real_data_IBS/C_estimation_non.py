
import numpy as np
import scipy.optimize as spo
from I_spline import I_U

def C_est_non(m, U, V, De1, De2, De3, g_X, nodevec):
    Iu = I_U(m, U * np.exp(g_X[:,0]), nodevec)
    Iv = I_U(m, V * np.exp(g_X[:,0]), nodevec)
    def LF(*args):
        a = args[0]
        Ezg = np.exp(g_X[:,1])
        Loss_F1 = - np.mean(De1 * np.log(1 - np.exp(- np.dot(Iu,a) * Ezg) + 1e-4) + De2 * np.log(np.exp(- np.dot(Iu,a) * Ezg) - np.exp(- np.dot(Iv,a) * Ezg) + 1e-4) - De3 * np.dot(Iv,a) * Ezg)
        return Loss_F1
    bnds = [(0, 100)] * (m+3)
    result = spo.minimize(LF, 0.1*np.ones(m+3), method='SLSQP', bounds=bnds)
    return result['x']


# def C_est_non(m, U, V, De1, De2, De3, g_X, nodevec, C0, eps=1e-12):
#     Iu = I_U(m, U * np.exp(g_X[:, 0]), nodevec) 
#     Iv = I_U(m, V * np.exp(g_X[:, 0]), nodevec)

#     def LF(a):
#         Ezg = np.exp(g_X[:, 1])
#         x_u = np.maximum(np.dot(Iu, a) * Ezg, 0.0)
#         x_v = np.maximum(np.dot(Iv, a) * Ezg, 0.0)

#         # s1 = log(1 - exp(-x_u)) = log1p(-exp(-x_u))
#         t1 = -np.exp(-x_u)
#         s1 = np.log1p(np.clip(t1, -1 + eps, -eps))  # t1 ∈ (-1, 0)

#         # s2 = log(exp(-x_u) - exp(-x_v)) = log(exp(-x_u) * (1 - exp(-(x_v-x_u))))
#         #    = -x_u + log1p(-exp(-(x_v - x_u)))
#         delta = x_v - x_u
#         t2 = -np.exp(-np.maximum(delta, 0.0))  # ensure within (-1,0)
#         s2 = -x_u + np.log1p(np.clip(t2, -1 + eps, -eps))

#         return -np.mean(De1 * s1 + De2 * s2 - De3 * x_v)

#     bnds = [(0.0, 100.0)] * (m + 3)
#     res = spo.minimize(LF, C0, method='SLSQP', bounds=bnds)
#     return res.x











