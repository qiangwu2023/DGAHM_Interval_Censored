# import numpy as np
# import scipy.optimize as spo
# from I_spline import I_U


# def gamma_est(beta, De1, De2, De3, Z, U, V, C, m, g_X, nodevec):
#     def BF(*args):
#         b = args[0]
#         Iu = I_U(m, U * np.exp(np.dot(Z, beta) + g_X[:,0]), nodevec)
#         Iv = I_U(m, V * np.exp(np.dot(Z, beta) + g_X[:,0]), nodevec)
#         Ezg = np.exp(np.dot(Z, b)+ g_X[:,1])
#         Loss_F = - np.mean(De1 * np.log(1 - np.exp(- np.dot(Iu, C) * Ezg) + 1e-4) + De2 * np.log(np.exp(- np.dot(Iu, C) * Ezg) - np.exp(- np.dot(Iv, C) * Ezg) + 1e-4) - De3 * np.dot(Iv, C) * Ezg)
#         return Loss_F
#     result = spo.minimize(BF,np.zeros(Z.shape[1]),method='SLSQP')
#     return result['x']




import numpy as np
import scipy.optimize as spo
from I_spline import I_U

def gamma_est(beta, De1, De2, De3, Z, U, V, C, m, g_X, nodevec,
              eps=1e-8, exp_cap=50.0):
    """
    稳健版本：通过裁剪防止 log(0)、负数取对数与指数溢出。
    eps:   对数的下限，避免 log(0)
    exp_cap: 指数输入的上/下界，避免溢出/下溢
    """

    # 转为数组，确保广播可靠
    Z = np.asarray(Z)
    g_X = np.asarray(g_X)
    U = np.asarray(U)
    V = np.asarray(V)
    C = np.asarray(C)
    De1 = np.asarray(De1)
    De2 = np.asarray(De2)
    De3 = np.asarray(De3)

    # 安全的 exp：对输入先截断
    def safe_exp(x):
        return np.exp(np.clip(x, -exp_cap, exp_cap))

    def BF(b):
        # 线性预测 for gamma
        Zb = Z @ b
        Zb = np.clip(Zb, -exp_cap, exp_cap)

        # 线性预测 for beta 已给定
        Zbeta = Z @ beta
        Zbeta = np.clip(Zbeta, -exp_cap, exp_cap)

        gx0 = np.clip(g_X[:, 0], -exp_cap, exp_cap)
        gx1 = np.clip(g_X[:, 1], -exp_cap, exp_cap)

        # 缩放 U, V
        scale_uv = safe_exp(Zbeta + gx0)
        U_scaled = U * scale_uv
        V_scaled = V * scale_uv

        # I-spline
        Iu = I_U(m, U_scaled, nodevec)   # (n, k)
        Iv = I_U(m, V_scaled, nodevec)   # (n, k)

        # Ezg
        Ezg = safe_exp(Zb + gx1)         # (n,)

        # 线性组合
        IuC = Iu @ C
        IvC = Iv @ C

        # 若理论要求非负，可裁剪到非负
        IuC = np.clip(IuC, 0.0, None)
        IvC = np.clip(IvC, 0.0, None)

        # lambda
        lam_u = IuC * Ezg
        lam_v = IvC * Ezg

        # 限制范围避免数值问题
        lam_u = np.clip(lam_u, 0.0, np.float64(exp_cap))
        lam_v = np.clip(lam_v, 0.0, np.float64(exp_cap))

        # 生存函数
        Su = np.exp(-lam_u)
        Sv = np.exp(-lam_v)

        # 三类项
        # term1: log(1 - Su)
        p1 = 1.0 - Su
        p1 = np.clip(p1, eps, 1.0)  # 防止 0
        term1 = De1 * np.log(p1)

        # term2: log(Su - Sv)
        diff = Su - Sv
        # 按理应非负；若因数值误差非正，裁剪为 eps
        diff = np.clip(diff, eps, 1.0)
        term2 = De2 * np.log(diff)

        # term3: -De3 * (IvC * Ezg)
        term3 = -De3 * lam_v

        Loss_F = -(np.mean(term1 + term2 + term3))

        if not np.isfinite(Loss_F):
            return 1e50
        return Loss_F

    x0 = np.zeros(Z.shape[1], dtype=float)
    result = spo.minimize(
        BF,
        x0,
        method='SLSQP',
        options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False}
    )
    return result['x']



