import numpy as np
import scipy.optimize as spo
from I_spline import I_U

def C_est(m, U, V, De1, De2, De3, g_X, nodevec, C0,
          eps=1e-8, exp_cap=50.0, uv_cap=1e6):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    g_X = np.asarray(g_X, dtype=float)
    De1 = np.asarray(De1, dtype=float)
    De2 = np.asarray(De2, dtype=float)
    De3 = np.asarray(De3, dtype=float)


    def safe_exp(x):
        return np.exp(np.clip(x, -exp_cap, exp_cap))


    gx0 = np.clip(g_X[:, 0], -exp_cap, exp_cap)
    scale_uv = safe_exp(gx0)

    U_scaled = np.clip(U * scale_uv, 0.0, uv_cap)
    V_scaled = np.clip(V * scale_uv, 0.0, uv_cap)

    Iu = I_U(m, U_scaled, nodevec) 
    Iv = I_U(m, V_scaled, nodevec)

    gx1 = np.clip(g_X[:, 1], -exp_cap, exp_cap)
    Ezg = safe_exp(gx1)


    def LF(a):
        IuA = Iu @ a
        IvA = Iv @ a

        IuA = np.clip(IuA, 0.0, np.float64(exp_cap))
        IvA = np.clip(IvA, 0.0, np.float64(exp_cap))

        lam_u = IuA * Ezg
        lam_v = IvA * Ezg

        lam_u = np.clip(lam_u, 0.0, np.float64(exp_cap))
        lam_v = np.clip(lam_v, 0.0, np.float64(exp_cap))

        Su = np.exp(-lam_u)
        Sv = np.exp(-lam_v)

        p1 = 1.0 - Su
        p1 = np.clip(p1, eps, 1.0)
        term1 = De1 * np.log(p1)

        diff = Su - Sv
        diff = np.clip(diff, eps, 1.0)
        term2 = De2 * np.log(diff)

        term3 = -De3 * lam_v

        Loss = -(np.mean(term1 + term2 + term3))
        if not np.isfinite(Loss):
            return 1e50
        return Loss

    bnds = [(0.0, 100.0)] * (m + 3)

    result = spo.minimize(
        LF,
        np.asarray(C0, dtype=float),
        method='SLSQP',
        bounds=bnds,
        options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False}
    )
    return result['x']
