import numpy as np

def indicate_T_t_matrix(t, S_t_W, S_U_W, S_V_W,
                        U_R_test, V_R_test, De1_R_test, De2_R_test, De3_R_test):
    m = len(t)
    n = len(U_R_test)
    out = np.zeros((m, n), dtype=float)

    for j in range(m):
        res = np.zeros(n, dtype=float)
        # 与单点 t 时的逻辑一致，但用 S_t_W[j, i]
        for i in range(n):
            if De1_R_test[i] == 1:
                if U_R_test[i] < t[j]:
                    res[i] = 0.0
                else:
                    res[i] = (S_t_W[j, i] - S_U_W[i]) / (1.0 - S_U_W[i])
            if De2_R_test[i] == 1:
                if t[j] < U_R_test[i]:
                    res[i] = 1.0
                elif t[j] >= V_R_test[i]:
                    res[i] = 0.0
                else:
                    res[i] = (S_t_W[j, i] - S_V_W[i]) / (S_U_W[i] - S_V_W[i])
            if De3_R_test[i] == 1:
                if t[j] < V_R_test[i]:
                    res[i] = 1.0
                else:
                    res[i] = S_t_W[j, i] / S_V_W[i]
        out[j] = res
    return out


# import numpy as np

# def indicate_T_t_matrix1(t, S_t_W, S_U_W, S_V_W,
#                             U_R_test, V_R_test, De1_R_test, De2_R_test, De3_R_test,
#                             eps=1e-12):
#     # t: (m,), S_t_W: (m, n); others: (n,)
#     t = np.asarray(t)
#     S_t_W = np.asarray(S_t_W)
#     S_U_W = np.asarray(S_U_W)
#     S_V_W = np.asarray(S_V_W)
#     U = np.asarray(U_R_test)
#     V = np.asarray(V_R_test)
#     De1 = np.asarray(De1_R_test).astype(bool)
#     De2 = np.asarray(De2_R_test).astype(bool)
#     De3 = np.asarray(De3_R_test).astype(bool)

#     m = t.shape[0]
#     n = U.shape[0]
#     out = np.zeros((m, n), dtype=float)

#     # 广播辅助：把 (n,) 扩为 (m, n)
#     T = t[:, None]              # (m,1)
#     U_mat = np.broadcast_to(U, (m, n))
#     V_mat = np.broadcast_to(V, (m, n))
#     S_U = np.broadcast_to(S_U_W, (m, n))
#     S_V = np.broadcast_to(S_V_W, (m, n))

#     # 掩码 (m, n) 根据 De* 按列复制
#     M1 = np.broadcast_to(De1, (m, n))
#     M2 = np.broadcast_to(De2, (m, n))
#     M3 = np.broadcast_to(De3, (m, n))

#     # De1 情形
#     idx1_a = M1 & (U_mat < T)                  # result = 0
#     idx1_b = M1 & ~(U_mat < T)                 # result = (S_t - S_U) / (1 - S_U)
#     denom1 = np.maximum(1.0 - S_U, eps)
#     out[idx1_b] = ((S_t_W - S_U)[idx1_b]) / denom1[idx1_b]
#     # idx1_a 已是 0，无需赋值

#     # De2 情形
#     idx2_a = M2 & (T < U_mat)                  # result = 1
#     idx2_b = M2 & (T >= V_mat)                 # result = 0
#     idx2_c = M2 & ~(idx2_a | idx2_b)           # U <= T < V
#     denom2 = np.maximum((S_U - S_V), eps)
#     out[idx2_a] = 1.0
#     # idx2_b 已是 0
#     out[idx2_c] = ((S_t_W - S_V)[idx2_c]) / denom2[idx2_c]

#     # De3 情形
#     idx3_a = M3 & (T < V_mat)                  # result = 1
#     idx3_b = M3 & ~(T < V_mat)                 # T >= V
#     denom3 = np.maximum(S_V, eps)
#     out[idx3_a] = 1.0
#     out[idx3_b] = (S_t_W[idx3_b]) / denom3[idx3_b]

#     return out
