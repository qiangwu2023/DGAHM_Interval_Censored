#%% ----------------------
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from iteration_deep_judge import Est_deep_judge
from iteration_linear_judge import Est_linear_judge
from iteration_deep import Est_deep
from iteration_linear import Est_linear
from Least_gamma import LFD_gamma
from I_spline import I_U
from I_spline import I_S
from B_spline2 import B_S2

#%% ---------- -------------
def set_seed(seed):
    np.random.seed(seed)  
    torch.manual_seed(seed) 
#%% ---------------------
set_seed(20)
#%% ----------- ------------
p = 3 
n_layer = 3 
n_node = 100 
n_epoch = 300 
n_lr = 5e-4 
m = 15 

#%% Data Processing
df = pd.read_csv('data_center.csv') 

Z = np.array(df[["BMI01","GLUCOS01","HDL01","TCHSIU01"]], dtype='float32')
X = np.array(df[["SBPA21","SBPA22","RACEGRP","GENDER","V1AGE01","Cen1","Cen2","Cen3","SBPA21_real","SBPA22_real","V1AGE01_real"]], dtype='float32')
U = np.array(df["U_year"], dtype='float32')
V = np.array(df["V_year"], dtype='float32')
De1 = np.array(df["De1"], dtype='float32')
De2 = np.array(df["De2"], dtype='float32')
De3 = np.array(df["De3"], dtype='float32')
np.min(X), np.max(X)

nodevec = np.array(np.linspace(0, 200, m+2), dtype="float32")
c_initial = np.array(0.5*np.ones(m+p), dtype="float32") 
theta_initial = np.array(np.zeros(2*Z.shape[1]), dtype='float32')

#%% =========================== 
A = np.arange(len(U))
np.random.shuffle(A)

Z_R = Z[A]
X_R = X[A]
U_R = U[A]
V_R = V[A]
De1_R = De1[A]
De2_R = De2[A]
De3_R = De3[A]

# ----------------------
# ---train data 11000
Z_R_train = Z_R[np.arange(11000)]
X_train_all = X_R[np.arange(11000)]
X_R_train = X_train_all[:,0:8]
U_R_train = U_R[np.arange(11000)]
V_R_train = V_R[np.arange(11000)]
De1_R_train = De1_R[np.arange(11000)]
De2_R_train = De2_R[np.arange(11000)]
De3_R_train = De3_R[np.arange(11000)]
# ---test data 2204
Z_R_test = Z_R[np.arange(11000,len(U))]
# Z_R_test = np.delete(Z_R, np.arange(11000), axis=0)
X_R_test = X_R[np.arange(11000,len(U))][:,0:8]
U_R_test = U_R[np.arange(11000,len(U))]
V_R_test = V_R[np.arange(11000,len(U))]
De1_R_test = De1_R[np.arange(11000,len(U))]
De2_R_test = De2_R[np.arange(11000,len(U))]
De3_R_test = De3_R[np.arange(11000,len(U))]


#%% Divide the samples of the test set into two classes（delta3=1, delta3=0）
# np.mean(De3_R_test) # right censoring rate
# --------- delta3 = 1----------
U_test1 = np.array(U_R_test[De3_R_test==1])
V_test1 = np.array(V_R_test[De3_R_test==1])
Z_test1 = np.array(Z_R_test[De3_R_test==1])
X_test1 = np.array(X_R_test[De3_R_test==1])
# Sort De_test1 to select 25%, 50%, 75% quantile points
X_sort1 = X_test1[V_test1.argsort()]
Z_sort1 = Z_test1[V_test1.argsort()]
U_sort1 = U_test1[V_test1.argsort()]
V_sort1 = V_test1[V_test1.argsort()]

n_V1 = len(V_sort1)
V1_025 = V_sort1[round(n_V1*0.25)]
V1_050 = V_sort1[round(n_V1*0.5)]
V1_075 = V_sort1[round(n_V1*0.75)]
V1 = [V1_025, V1_050, V1_075]
# Draw horizontal coordinate of the graph with delta=1
V1_value = np.array(np.linspace(0, 30, 20), dtype="float32")


# --------- delta3 = 0----------
U_test0 = np.array(U_R_test[De3_R_test==0])
V_test0 = np.array(V_R_test[De3_R_test==0])
Z_test0 = np.array(Z_R_test[De3_R_test==0])
X_test0 = np.array(X_R_test[De3_R_test==0])
# Sort De_test0 to select 25%, 50%, 75% quantile points
X_sort0 = X_test0[V_test0.argsort()]
Z_sort0 = Z_test0[V_test0.argsort()]
U_sort0 = U_test0[V_test0.argsort()]
V_sort0 = V_test0[V_test0.argsort()]

n_V0 = len(V_sort0)
V0_025 = V_sort0[round(n_V0*0.25)]
V0_050 = V_sort0[round(n_V0*0.5)]
V0_075 = V_sort0[round(n_V0*0.75)]
V0 = [V0_025, V0_050, V0_075]
# Draw horizontal coordinate of the graph with delta=0
V0_value = np.array(np.linspace(0, 30, 20), dtype="float32")

#%% =========================
X_class = X_train_all[(X_train_all[:,2]==0)*(X_train_all[:,3]==1)*(X_train_all[:,5]==0)*(X_train_all[:,6]==0)*(X_train_all[:,7]==1)]
X_mean = np.mean(X_class[:,0:8], axis=0)
X_data1 = np.tile(X_mean, (100,1))
Min1 = np.min(X_class[:,0])
Max1 = np.max(X_class[:,0])
x_value1 = np.array(np.linspace(Min1, Max1, 100), dtype="float32") 
X_data1[:,0] = x_value1
X1_min = np.min(X_class[:,8])
X1_max = np.max(X_class[:,8])
X_value1 = np.array(np.linspace(X1_min, X1_max, 100), dtype="float32") 

X_data2 = np.tile(X_mean, (100,1))
Min2 = np.min(X_class[:,1])
Max2 = np.max(X_class[:,1])
x_value2 = np.array(np.linspace(Min2, Max2, 100), dtype="float32") 
X_data2[:,1] = x_value2
X2_min = np.min(X_class[:,9])
X2_max = np.max(X_class[:,9])
X_value2 = np.array(np.linspace(X2_min, X2_max, 100), dtype="float32") 

X_data3 = np.tile(X_mean, (100,1))
Min3 = np.min(X_class[:,4])
Max3 = np.max(X_class[:,4])
x_value3 = np.array(np.linspace(Min3, Max3, 100), dtype="float32") 
X_data3[:,4] = x_value3
X3_min = np.min(X_class[:,10])
X3_max = np.max(X_class[:,10])
X_value3 = np.array(np.linspace(X3_min, X3_max, 100), dtype="float32") 

fig_g_x1 = plt.figure()
ax_g_x1 = fig_g_x1.add_subplot(1, 1, 1)
plt.xlim(X1_min,X1_max) 
ax_g_x1.set_xlabel("Systolic blood pressure",fontsize=8)       
ax_g_x1.set_ylabel(r"The value of $g$",fontsize=8) 
ax_g_x1.tick_params(axis='both',labelsize=6) 
# ----------------------------------
fig_g_x2 = plt.figure()
ax_g_x2 = fig_g_x2.add_subplot(1, 1, 1)
plt.xlim(X2_min,X2_max) 
ax_g_x2.set_xlabel("Diastolic blood pressure",fontsize=8)       
ax_g_x2.set_ylabel(r"The value of $g$",fontsize=8) 
ax_g_x2.tick_params(axis='both',labelsize=6) 
# ----------------------------------
fig_g_x3 = plt.figure()
ax_g_x3 = fig_g_x3.add_subplot(1, 1, 1)
plt.xlim(X3_min,X3_max) 
ax_g_x3.set_xlabel("Age",fontsize=8)       
ax_g_x3.set_ylabel(r"The value of $g$",fontsize=8) 
ax_g_x3.tick_params(axis='both',labelsize=6) 

# ----------------------
fig_g_diff = plt.figure()
ax_g_diff = fig_g_diff.add_subplot(1, 1, 1)
ax_g_diff.set_xlabel("subject",fontsize=8)       
ax_g_diff.set_ylabel(r"The difference of $g$",fontsize=8) 
ax_g_diff.tick_params(axis='both',labelsize=6) 

# ---------------------
fig_g_deep = plt.figure()
ax_g_deep = fig_g_deep.add_subplot(1, 1, 1)
ax_g_deep.set_title('(a)', fontsize=10) 
ax_g_deep.set_xlabel("subject",fontsize=8)       
ax_g_deep.set_ylabel(r"The estimate of $g$",fontsize=8) 
ax_g_deep.tick_params(axis='both',labelsize=6) 

# --------------------
fig_g_deep_no = plt.figure()
ax_g_deep_no = fig_g_deep_no.add_subplot(1, 1, 1)
ax_g_deep_no.set_xlabel("subject",fontsize=8)       
ax_g_deep_no.set_ylabel(r"The estimate of $g$",fontsize=8) 
ax_g_deep_no.tick_params(axis='both',labelsize=6) 

# ----------------------
fig_g_Cox = plt.figure()
ax_g_Cox = fig_g_Cox.add_subplot(1, 1, 1)
ax_g_Cox.set_title('(b)', fontsize=10) 
ax_g_Cox.set_xlabel("subject",fontsize=8)  
ax_g_Cox.set_ylabel(r"The estimate of $g$",fontsize=8) 
ax_g_Cox.tick_params(axis='both',labelsize=6) 

fig_g_Cox_no = plt.figure()
ax_g_Cox_no = fig_g_Cox_no.add_subplot(1, 1, 1)
ax_g_Cox_no.set_xlabel("subject",fontsize=8)  
ax_g_Cox_no.set_ylabel(r"The estimate of $g$",fontsize=8) 
ax_g_Cox_no.tick_params(axis='both',labelsize=6) 

# -----------------------------
# ------------deep-------------
Est_Deep = Est_deep_judge(Z_R_train,X_R_train,U_R_train,V_R_train,De1_R_train,De2_R_train,De3_R_train,X_data1,X_data2,X_data3,theta_initial,n_layer,n_node,n_lr,n_epoch,nodevec,m,c_initial)
theta_Deep = Est_Deep['theta']
Ezg1_Deep = np.exp(np.dot(Z_R_train, theta_Deep[0:4]) + Est_Deep['g_train'][:,0])
Ezg2_Deep = np.exp(np.dot(Z_R_train, theta_Deep[4:8]) + Est_Deep['g_train'][:,1])
Iu_Deep = I_U(m, U_R_train * Ezg1_Deep, nodevec)
Iv_Deep = I_U(m, V_R_train * Ezg1_Deep, nodevec)
Lamb_U_Deep = np.dot(Iu_Deep, Est_Deep['C'])
Lamb_V_Deep = np.dot(Iv_Deep, Est_Deep['C'])
f_U_Deep = Lamb_U_Deep * Ezg2_Deep
f_V_Deep = Lamb_V_Deep * Ezg2_Deep
Bu_Deep = B_S2(m, U_R_train * Ezg1_Deep, nodevec)
Bv_Deep = B_S2(m, V_R_train * Ezg1_Deep, nodevec)
dLamb_U_Deep = np.matmul(Bu_Deep, Est_Deep['C'])
dLamb_V_Deep = np.matmul(Bv_Deep, Est_Deep['C'])
abc_gamma_Deep1 = LFD_gamma(Z_R_train[:,0],Z_R_train,X_R_train,U_R_train,V_R_train,De1_R_train,De2_R_train,De3_R_train,Est_Deep['g_train'],Est_Deep['theta'],Est_Deep['C'],m,nodevec,n_layer,n_node=50,n_lr=1e-4,n_epoch=200)

L_gamma_Deep1 = De1_R_train * Z_R_train[:,0] * f_U_Deep * np.exp(-f_U_Deep) / (1 - np.exp(-f_U_Deep) + 1e-4) + De2_R_train * Z_R_train[:,0] * (f_V_Deep * np.exp(-f_V_Deep)- f_U_Deep * np.exp(-f_U_Deep)) / (np.exp(-f_U_Deep) - np.exp(-f_V_Deep) + 1e-4) - De3_R_train * Z_R_train[:,0] * f_V_Deep

L_g1_gamma_Deep1 = (De1_R_train * U_R_train * dLamb_U_Deep * Ezg1_Deep * Ezg2_Deep * np.exp(-f_U_Deep) / (1 - np.exp(-f_U_Deep) + 1e-4) + De2_R_train * Ezg1_Deep * Ezg2_Deep * (V_R_train * dLamb_V_Deep * np.exp(-f_V_Deep) - U_R_train * dLamb_U_Deep * np.exp(-f_U_Deep)) / (np.exp(-f_U_Deep) - np.exp(-f_V_Deep) + 1e-4) - De3_R_train * V_R_train * dLamb_V_Deep * Ezg1_Deep * Ezg2_Deep) * abc_gamma_Deep1[:,0]

L_g2_gamma_Deep1 = (De1_R_train * f_U_Deep * np.exp(-f_U_Deep) / (1 - np.exp(-f_U_Deep) + 1e-4) + De2_R_train * (f_V_Deep * np.exp(-f_V_Deep)- f_U_Deep * np.exp(-f_U_Deep)) / (np.exp(-f_U_Deep) - np.exp(-f_V_Deep) + 1e-4) - De3_R_train * f_V_Deep) * abc_gamma_Deep1[:,1] 

L_lambda_gamma_D1 = De1_R_train * f_U_Deep * np.exp(-f_U_Deep) * abc_gamma_Deep1[:,2] / (1 - np.exp(-f_U_Deep) + 1e-4) + De2_R_train * (abc_gamma_Deep1[:,3] * f_V_Deep * np.exp(-f_V_Deep)- abc_gamma_Deep1[:,2] * f_U_Deep * np.exp(-f_U_Deep)) / (np.exp(-f_U_Deep) - np.exp(-f_V_Deep) + 1e-4) - De3_R_train * f_V_Deep * abc_gamma_Deep1[:,3]

I_1_Deep = L_gamma_Deep1 - L_g1_gamma_Deep1 - L_g2_gamma_Deep1 - L_lambda_gamma_D1

abc_gamma_Deep2 = LFD_gamma(Z_R_train[:,1],Z_R_train,X_R_train,U_R_train,V_R_train,De1_R_train,De2_R_train,De3_R_train,Est_Deep['g_train'],Est_Deep['theta'],Est_Deep['C'],m,nodevec,n_layer,n_node=50,n_lr=1e-4,n_epoch=200)

L_gamma_Deep2 = De1_R_train * Z_R_train[:,1] * f_U_Deep * np.exp(-f_U_Deep) / (1 - np.exp(-f_U_Deep) + 1e-4) + De2_R_train * Z_R_train[:,1] * (f_V_Deep * np.exp(-f_V_Deep)- f_U_Deep * np.exp(-f_U_Deep)) / (np.exp(-f_U_Deep) - np.exp(-f_V_Deep) + 1e-4) - De3_R_train * Z_R_train[:,1] * f_V_Deep

L_g1_gamma_Deep2 = (De1_R_train * U_R_train * dLamb_U_Deep * Ezg1_Deep * Ezg2_Deep * np.exp(-f_U_Deep) / (1 - np.exp(-f_U_Deep) + 1e-4) + De2_R_train * Ezg1_Deep * Ezg2_Deep * (V_R_train * dLamb_V_Deep * np.exp(-f_V_Deep) - U_R_train * dLamb_U_Deep * np.exp(-f_U_Deep)) / (np.exp(-f_U_Deep) - np.exp(-f_V_Deep) + 1e-4) - De3_R_train * V_R_train * dLamb_V_Deep * Ezg1_Deep * Ezg2_Deep) * abc_gamma_Deep2[:,0]

L_g2_gamma_Deep2 = (De1_R_train * f_U_Deep * np.exp(-f_U_Deep) / (1 - np.exp(-f_U_Deep) + 1e-4) + De2_R_train * (f_V_Deep * np.exp(-f_V_Deep)- f_U_Deep * np.exp(-f_U_Deep)) / (np.exp(-f_U_Deep) - np.exp(-f_V_Deep) + 1e-4) - De3_R_train * f_V_Deep) * abc_gamma_Deep2[:,1] 

L_lambda_gamma_Deep2 = De1_R_train * f_U_Deep * np.exp(-f_U_Deep) * abc_gamma_Deep2[:,2] / (1 - np.exp(-f_U_Deep) + 1e-4) + De2_R_train * (abc_gamma_Deep2[:,3] * f_V_Deep * np.exp(-f_V_Deep)- abc_gamma_Deep2[:,2] * f_U_Deep * np.exp(-f_U_Deep)) / (np.exp(-f_U_Deep) - np.exp(-f_V_Deep) + 1e-4) - De3_R_train * f_V_Deep * abc_gamma_Deep2[:,3]

I_2_Deep = L_gamma_Deep2 - L_g1_gamma_Deep2 - L_g2_gamma_Deep2 - L_lambda_gamma_Deep2

abc_gamma_Deep3 = LFD_gamma(Z_R_train[:,2],Z_R_train,X_R_train,U_R_train,V_R_train,De1_R_train,De2_R_train,De3_R_train,Est_Deep['g_train'],Est_Deep['theta'],Est_Deep['C'],m,nodevec,n_layer,n_node=50,n_lr=1e-4,n_epoch=200)

L_gamma_Deep3 = De1_R_train * Z_R_train[:,2] * f_U_Deep * np.exp(-f_U_Deep) / (1 - np.exp(-f_U_Deep) + 1e-4) + De2_R_train * Z_R_train[:,2] * (f_V_Deep * np.exp(-f_V_Deep)- f_U_Deep * np.exp(-f_U_Deep)) / (np.exp(-f_U_Deep) - np.exp(-f_V_Deep) + 1e-4) - De3_R_train * Z_R_train[:,2] * f_V_Deep

L_g1_gamma_Deep3 = (De1_R_train * U_R_train * dLamb_U_Deep * Ezg1_Deep * Ezg2_Deep * np.exp(-f_U_Deep) / (1 - np.exp(-f_U_Deep) + 1e-4) + De2_R_train * Ezg1_Deep * Ezg2_Deep * (V_R_train * dLamb_V_Deep * np.exp(-f_V_Deep) - U_R_train * dLamb_U_Deep * np.exp(-f_U_Deep)) / (np.exp(-f_U_Deep) - np.exp(-f_V_Deep) + 1e-4) - De3_R_train * V_R_train * dLamb_V_Deep * Ezg1_Deep * Ezg2_Deep) * abc_gamma_Deep3[:,0]

L_g2_gamma_Deep3 = (De1_R_train * f_U_Deep * np.exp(-f_U_Deep) / (1 - np.exp(-f_U_Deep) + 1e-4) + De2_R_train * (f_V_Deep * np.exp(-f_V_Deep)- f_U_Deep * np.exp(-f_U_Deep)) / (np.exp(-f_U_Deep) - np.exp(-f_V_Deep) + 1e-4) - De3_R_train * f_V_Deep) * abc_gamma_Deep3[:,1] 

L_lambda_gamma_Deep3 = De1_R_train * f_U_Deep * np.exp(-f_U_Deep) * abc_gamma_Deep3[:,2] / (1 - np.exp(-f_U_Deep) + 1e-4) + De2_R_train * (abc_gamma_Deep3[:,3] * f_V_Deep * np.exp(-f_V_Deep)- abc_gamma_Deep3[:,2] * f_U_Deep * np.exp(-f_U_Deep)) / (np.exp(-f_U_Deep) - np.exp(-f_V_Deep) + 1e-4) - De3_R_train * f_V_Deep * abc_gamma_Deep3[:,3]

I_3_Deep = L_gamma_Deep3 - L_g1_gamma_Deep3 - L_g2_gamma_Deep3 - L_lambda_gamma_Deep3

abc_gamma_Deep4 = LFD_gamma(Z_R_train[:,3],Z_R_train,X_R_train,U_R_train,V_R_train,De1_R_train,De2_R_train,De3_R_train,Est_Deep['g_train'],Est_Deep['theta'],Est_Deep['C'],m,nodevec,n_layer,n_node=50,n_lr=1e-4,n_epoch=200)

L_gamma_Deep4 = De1_R_train * Z_R_train[:,3] * f_U_Deep * np.exp(-f_U_Deep) / (1 - np.exp(-f_U_Deep) + 1e-4) + De2_R_train * Z_R_train[:,3] * (f_V_Deep * np.exp(-f_V_Deep)- f_U_Deep * np.exp(-f_U_Deep)) / (np.exp(-f_U_Deep) - np.exp(-f_V_Deep) + 1e-4) - De3_R_train * Z_R_train[:,3] * f_V_Deep

L_g1_gamma_Deep4 = (De1_R_train * U_R_train * dLamb_U_Deep * Ezg1_Deep * Ezg2_Deep * np.exp(-f_U_Deep) / (1 - np.exp(-f_U_Deep) + 1e-4) + De2_R_train * Ezg1_Deep * Ezg2_Deep * (V_R_train * dLamb_V_Deep * np.exp(-f_V_Deep) - U_R_train * dLamb_U_Deep * np.exp(-f_U_Deep)) / (np.exp(-f_U_Deep) - np.exp(-f_V_Deep) + 1e-4) - De3_R_train * V_R_train * dLamb_V_Deep * Ezg1_Deep * Ezg2_Deep) * abc_gamma_Deep4[:,0]

L_g2_gamma_Deep4 = (De1_R_train * f_U_Deep * np.exp(-f_U_Deep) / (1 - np.exp(-f_U_Deep) + 1e-4) + De2_R_train * (f_V_Deep * np.exp(-f_V_Deep)- f_U_Deep * np.exp(-f_U_Deep)) / (np.exp(-f_U_Deep) - np.exp(-f_V_Deep) + 1e-4) - De3_R_train * f_V_Deep) * abc_gamma_Deep4[:,1] 

L_lambda_gamma_Deep4 = De1_R_train * f_U_Deep * np.exp(-f_U_Deep) * abc_gamma_Deep4[:,2] / (1 - np.exp(-f_U_Deep) + 1e-4) + De2_R_train * (abc_gamma_Deep4[:,3] * f_V_Deep * np.exp(-f_V_Deep)- abc_gamma_Deep4[:,2] * f_U_Deep * np.exp(-f_U_Deep)) / (np.exp(-f_U_Deep) - np.exp(-f_V_Deep) + 1e-4) - De3_R_train * f_V_Deep * abc_gamma_Deep4[:,3]

I_4_Deep = L_gamma_Deep4 - L_g1_gamma_Deep4 - L_g2_gamma_Deep4 - L_lambda_gamma_Deep4

Info_Deep = np.zeros((4,4))
Info_Deep[0,0] = np.mean(I_1_Deep**2)
Info_Deep[1,1] = np.mean(I_2_Deep**2)
Info_Deep[2,2] = np.mean(I_3_Deep**2)
Info_Deep[3,3] = np.mean(I_4_Deep**2)
Info_Deep[0,1] = np.mean(I_1_Deep*I_2_Deep)
Info_Deep[1,0] = Info_Deep[0,1]
Info_Deep[0,2] = np.mean(I_1_Deep*I_3_Deep)
Info_Deep[2,0] = Info_Deep[0,2]
Info_Deep[0,3] = np.mean(I_1_Deep*I_4_Deep)
Info_Deep[3,0] = Info_Deep[0,3]
Info_Deep[1,2] = np.mean(I_2_Deep*I_3_Deep)
Info_Deep[2,1] = Info_Deep[1,2]
Info_Deep[1,3] = np.mean(I_2_Deep*I_4_Deep)
Info_Deep[3,1] = Info_Deep[1,3]
Info_Deep[2,3] = np.mean(I_3_Deep*I_4_Deep)
Info_Deep[3,2] = Info_Deep[2,3]
Sigma_Deep = np.linalg.inv(Info_Deep)/len(U_R_train)
sd1_Deep = np.sqrt(Sigma_Deep[0,0])
sd2_Deep = np.sqrt(Sigma_Deep[1,1])
sd3_Deep = np.sqrt(Sigma_Deep[2,2])
sd4_Deep = np.sqrt(Sigma_Deep[3,3])
dic_D = {"gamma_deep": theta_Deep[4:8], "sd_deep": np.sqrt(np.diag(Sigma_Deep))}
Result_deep = pd.DataFrame(dic_D,index=['gamma1','gamma2','gamma3','gamma4'])
Result_deep.to_csv('Result_deep.csv')
# -----------g_x-----------------
ax_g_x1.plot(X_value1, Est_Deep['g_value1'][:,1])
ax_g_x2.plot(X_value2, Est_Deep['g_value2'][:,1])
ax_g_x3.plot(X_value3, Est_Deep['g_value3'][:,1])

fig_g_x1.savefig('fig_g_x1.jpeg', dpi=400, bbox_inches='tight')
fig_g_x2.savefig('fig_g_x2.jpeg', dpi=400, bbox_inches='tight')
fig_g_x3.savefig('fig_g_x3.jpeg', dpi=400, bbox_inches='tight')





# ------------Cox--------------
Z_new = np.array(df[["BMI01","GLUCOS01","HDL01","TCHSIU01","V1AGE01","SBPA21","SBPA22"]], dtype='float32')[A][np.arange(11000)]
X_new = np.array(df[["RACEGRP","GENDER","Cen1","Cen2","Cen3"]], dtype='float32')[A][np.arange(11000)]
theta0_new = np.array(0.5*np.ones(2*Z_new.shape[1]), dtype='float32')
Est_Linear = Est_linear_judge(Z_new,X_new,U_R_train,V_R_train,De1_R_train,De2_R_train,De3_R_train,theta0_new,nodevec,m,c_initial)
theta_Linear = Est_Linear['theta']
g_other5 = Est_Linear['g_parameter']
n_L = Z_new.shape[1]
Ezg1_L = np.exp(np.dot(Z_new, theta_Linear[0:n_L]) + Est_Linear['g_train'][:,0])
Ezg2_L = np.exp(np.dot(Z_new, theta_Linear[n_L:(2*n_L)]) + Est_Linear['g_train'][:,1])
Iu_L = I_U(m, U_R_train * Ezg1_L, nodevec)
Iv_L = I_U(m, V_R_train * Ezg1_L, nodevec)
Lamb_U_L = np.dot(Iu_L, Est_Linear['C'])
Lamb_V_L = np.dot(Iv_L, Est_Linear['C'])
f_U_L = Lamb_U_L * Ezg2_L
f_V_L = Lamb_V_L * Ezg2_L
Bu_L = B_S2(m, U_R_train * Ezg1_L, nodevec)
Bv_L = B_S2(m, V_R_train * Ezg1_L, nodevec)
dLamb_U_L = np.matmul(Bu_L, Est_Linear['C'])
dLamb_V_L = np.matmul(Bv_L, Est_Linear['C'])
abc_gamma_L1 = LFD_gamma(Z_new[:,0],Z_new,X_new,U_R_train,V_R_train,De1_R_train,De2_R_train,De3_R_train,Est_Linear['g_train'],Est_Linear['theta'],Est_Linear['C'],m,nodevec,n_layer,n_node=50,n_lr=1e-4,n_epoch=200)

L_gamma_L1 = De1_R_train * Z_new[:,0] * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_R_train * Z_new[:,0] * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_R_train * Z_new[:,0] * f_V_L

L_g1_gamma_L1 = (De1_R_train * U_R_train * dLamb_U_L * Ezg1_L * Ezg2_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_R_train * Ezg1_L * Ezg2_L * (V_R_train * dLamb_V_L * np.exp(-f_V_L) - U_R_train * dLamb_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_R_train * V_R_train * dLamb_V_L * Ezg1_L * Ezg2_L) * abc_gamma_L1[:,0]

L_g2_gamma_L1 = (De1_R_train * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_R_train * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_R_train * f_V_L) * abc_gamma_L1[:,1] 

L_lambda_gamma_L1 = De1_R_train * f_U_L * np.exp(-f_U_L) * abc_gamma_L1[:,2] / (1 - np.exp(-f_U_L) + 1e-4) + De2_R_train * (abc_gamma_L1[:,3] * f_V_L * np.exp(-f_V_L)- abc_gamma_L1[:,2] * f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_R_train * f_V_L * abc_gamma_L1[:,3]

I_1_Linear = L_gamma_L1 - L_g1_gamma_L1 - L_g2_gamma_L1 - L_lambda_gamma_L1

abc_gamma_L2 = LFD_gamma(Z_new[:,1],Z_new,X_new,U_R_train,V_R_train,De1_R_train,De2_R_train,De3_R_train,Est_Linear['g_train'],Est_Linear['theta'],Est_Linear['C'],m,nodevec,n_layer,n_node=50,n_lr=1e-4,n_epoch=200)

L_gamma_L2 = De1_R_train * Z_new[:,1] * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_R_train * Z_new[:,1] * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_R_train * Z_new[:,1] * f_V_L

L_g1_gamma_L2 = (De1_R_train * U_R_train * dLamb_U_L * Ezg1_L * Ezg2_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_R_train * Ezg1_L * Ezg2_L * (V_R_train * dLamb_V_L * np.exp(-f_V_L) - U_R_train * dLamb_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_R_train * V_R_train * dLamb_V_L * Ezg1_L * Ezg2_L) * abc_gamma_L2[:,0]

L_g2_gamma_L2 = (De1_R_train * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_R_train * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_R_train * f_V_L) * abc_gamma_L2[:,1] 

L_lambda_gamma_L2 = De1_R_train * f_U_L * np.exp(-f_U_L) * abc_gamma_L2[:,2] / (1 - np.exp(-f_U_L) + 1e-4) + De2_R_train * (abc_gamma_L2[:,3] * f_V_L * np.exp(-f_V_L)- abc_gamma_L2[:,2] * f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_R_train * f_V_L * abc_gamma_L2[:,3]

I_2_Linear = L_gamma_L2 - L_g1_gamma_L2 - L_g2_gamma_L2 - L_lambda_gamma_L2

abc_gamma_L3 = LFD_gamma(Z_new[:,2],Z_new,X_new,U_R_train,V_R_train,De1_R_train,De2_R_train,De3_R_train,Est_Linear['g_train'],Est_Linear['theta'],Est_Linear['C'],m,nodevec,n_layer,n_node=50,n_lr=1e-4,n_epoch=200)

L_gamma_L3 = De1_R_train * Z_new[:,2] * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_R_train * Z_new[:,2] * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_R_train * Z_new[:,2] * f_V_L

L_g1_gamma_L3 = (De1_R_train * U_R_train * dLamb_U_L * Ezg1_L * Ezg2_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_R_train * Ezg1_L * Ezg2_L * (V_R_train * dLamb_V_L * np.exp(-f_V_L) - U_R_train * dLamb_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_R_train * V_R_train * dLamb_V_L * Ezg1_L * Ezg2_L) * abc_gamma_L3[:,0]

L_g2_gamma_L3 = (De1_R_train * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_R_train * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_R_train * f_V_L) * abc_gamma_L3[:,1] 

L_lambda_gamma_L3 = De1_R_train * f_U_L * np.exp(-f_U_L) * abc_gamma_L3[:,2] / (1 - np.exp(-f_U_L) + 1e-4) + De2_R_train * (abc_gamma_L3[:,3] * f_V_L * np.exp(-f_V_L)- abc_gamma_L3[:,2] * f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_R_train * f_V_L * abc_gamma_L3[:,3]

I_3_Linear = L_gamma_L3 - L_g1_gamma_L3 - L_g2_gamma_L3 - L_lambda_gamma_L3

abc_gamma_L4 = LFD_gamma(Z_new[:,3],Z_new,X_new,U_R_train,V_R_train,De1_R_train,De2_R_train,De3_R_train,Est_Linear['g_train'],Est_Linear['theta'],Est_Linear['C'],m,nodevec,n_layer,n_node=50,n_lr=1e-4,n_epoch=200)

L_gamma_L4 = De1_R_train * Z_new[:,3] * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_R_train * Z_new[:,3] * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_R_train * Z_new[:,3] * f_V_L

L_g1_gamma_L4 = (De1_R_train * U_R_train * dLamb_U_L * Ezg1_L * Ezg2_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_R_train * Ezg1_L * Ezg2_L * (V_R_train * dLamb_V_L * np.exp(-f_V_L) - U_R_train * dLamb_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_R_train * V_R_train * dLamb_V_L * Ezg1_L * Ezg2_L) * abc_gamma_L4[:,0]

L_g2_gamma_L4 = (De1_R_train * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_R_train * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_R_train * f_V_L) * abc_gamma_L4[:,1] 

L_lambda_gamma_L4 = De1_R_train * f_U_L * np.exp(-f_U_L) * abc_gamma_L4[:,2] / (1 - np.exp(-f_U_L) + 1e-4) + De2_R_train * (abc_gamma_L4[:,3] * f_V_L * np.exp(-f_V_L)- abc_gamma_L4[:,2] * f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_R_train * f_V_L * abc_gamma_L4[:,3]

I_4_Linear = L_gamma_L4 - L_g1_gamma_L4 - L_g2_gamma_L4 - L_lambda_gamma_L4

abc_gamma_L5 = LFD_gamma(Z_new[:,4],Z_new,X_new,U_R_train,V_R_train,De1_R_train,De2_R_train,De3_R_train,Est_Linear['g_train'],Est_Linear['theta'],Est_Linear['C'],m,nodevec,n_layer,n_node=50,n_lr=1e-4,n_epoch=200)

L_gamma_L5 = De1_R_train * Z_new[:,3] * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_R_train * Z_new[:,3] * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_R_train * Z_new[:,3] * f_V_L

L_g1_gamma_L5 = (De1_R_train * U_R_train * dLamb_U_L * Ezg1_L * Ezg2_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_R_train * Ezg1_L * Ezg2_L * (V_R_train * dLamb_V_L * np.exp(-f_V_L) - U_R_train * dLamb_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_R_train * V_R_train * dLamb_V_L * Ezg1_L * Ezg2_L) * abc_gamma_L5[:,0]

L_g2_gamma_L5 = (De1_R_train * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_R_train * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_R_train * f_V_L) * abc_gamma_L5[:,1] 

L_lambda_gamma_L5 = De1_R_train * f_U_L * np.exp(-f_U_L) * abc_gamma_L5[:,2] / (1 - np.exp(-f_U_L) + 1e-4) + De2_R_train * (abc_gamma_L5[:,3] * f_V_L * np.exp(-f_V_L)- abc_gamma_L5[:,2] * f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_R_train * f_V_L * abc_gamma_L5[:,3]

I_5_Linear = L_gamma_L5 - L_g1_gamma_L5 - L_g2_gamma_L5 - L_lambda_gamma_L5

abc_gamma_L6 = LFD_gamma(Z_new[:,5],Z_new,X_new,U_R_train,V_R_train,De1_R_train,De2_R_train,De3_R_train,Est_Linear['g_train'],Est_Linear['theta'],Est_Linear['C'],m,nodevec,n_layer,n_node=50,n_lr=1e-4,n_epoch=200)

L_gamma_L6 = De1_R_train * Z_new[:,3] * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_R_train * Z_new[:,3] * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_R_train * Z_new[:,3] * f_V_L

L_g1_gamma_L6 = (De1_R_train * U_R_train * dLamb_U_L * Ezg1_L * Ezg2_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_R_train * Ezg1_L * Ezg2_L * (V_R_train * dLamb_V_L * np.exp(-f_V_L) - U_R_train * dLamb_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_R_train * V_R_train * dLamb_V_L * Ezg1_L * Ezg2_L) * abc_gamma_L6[:,0]

L_g2_gamma_L6 = (De1_R_train * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_R_train * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_R_train * f_V_L) * abc_gamma_L6[:,1] 

L_lambda_gamma_L6 = De1_R_train * f_U_L * np.exp(-f_U_L) * abc_gamma_L6[:,2] / (1 - np.exp(-f_U_L) + 1e-4) + De2_R_train * (abc_gamma_L6[:,3] * f_V_L * np.exp(-f_V_L)- abc_gamma_L6[:,2] * f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_R_train * f_V_L * abc_gamma_L6[:,3]

I_6_Linear = L_gamma_L6 - L_g1_gamma_L6 - L_g2_gamma_L6 - L_lambda_gamma_L6

abc_gamma_L7 = LFD_gamma(Z_new[:,5],Z_new,X_new,U_R_train,V_R_train,De1_R_train,De2_R_train,De3_R_train,Est_Linear['g_train'],Est_Linear['theta'],Est_Linear['C'],m,nodevec,n_layer,n_node=50,n_lr=1e-4,n_epoch=200)

L_gamma_L7 = De1_R_train * Z_new[:,3] * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_R_train * Z_new[:,3] * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_R_train * Z_new[:,3] * f_V_L

L_g1_gamma_L7 = (De1_R_train * U_R_train * dLamb_U_L * Ezg1_L * Ezg2_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_R_train * Ezg1_L * Ezg2_L * (V_R_train * dLamb_V_L * np.exp(-f_V_L) - U_R_train * dLamb_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_R_train * V_R_train * dLamb_V_L * Ezg1_L * Ezg2_L) * abc_gamma_L7[:,0]

L_g2_gamma_L7 = (De1_R_train * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_R_train * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_R_train * f_V_L) * abc_gamma_L7[:,1] 

L_lambda_gamma_L7 = De1_R_train * f_U_L * np.exp(-f_U_L) * abc_gamma_L7[:,2] / (1 - np.exp(-f_U_L) + 1e-4) + De2_R_train * (abc_gamma_L7[:,3] * f_V_L * np.exp(-f_V_L)- abc_gamma_L7[:,2] * f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_R_train * f_V_L * abc_gamma_L7[:,3]

I_7_Linear = L_gamma_L7 - L_g1_gamma_L7 - L_g2_gamma_L7 - L_lambda_gamma_L7


Info_Linear = np.zeros((7,7))
Info_Linear[0,0] = np.mean(I_1_Linear**2)
Info_Linear[1,1] = np.mean(I_2_Linear**2)
Info_Linear[2,2] = np.mean(I_3_Linear**2)
Info_Linear[3,3] = np.mean(I_4_Linear**2)
Info_Linear[4,4] = np.mean(I_5_Linear**2)
Info_Linear[5,5] = np.mean(I_6_Linear**2)
Info_Linear[6,6] = np.mean(I_7_Linear**2)
Info_Linear[0,1] = np.mean(I_1_Linear*I_2_Linear)
Info_Linear[1,0] = Info_Linear[0,1]
Info_Linear[0,2] = np.mean(I_1_Linear*I_3_Linear)
Info_Linear[2,0] = Info_Linear[0,2]
Info_Linear[0,3] = np.mean(I_1_Linear*I_4_Linear)
Info_Linear[3,0] = Info_Linear[0,3]
Info_Linear[0,4] = np.mean(I_1_Linear*I_5_Linear)
Info_Linear[4,0] = Info_Linear[0,4]
Info_Linear[0,5] = np.mean(I_1_Linear*I_6_Linear)
Info_Linear[5,0] = Info_Linear[0,5]
Info_Linear[0,6] = np.mean(I_1_Linear*I_7_Linear)
Info_Linear[6,0] = Info_Linear[0,6]
Info_Linear[1,2] = np.mean(I_2_Linear*I_3_Linear)
Info_Linear[2,1] = Info_Linear[1,2]
Info_Linear[1,3] = np.mean(I_2_Linear*I_4_Linear)
Info_Linear[3,1] = Info_Linear[1,3]
Info_Linear[1,4] = np.mean(I_2_Linear*I_5_Linear)
Info_Linear[4,1] = Info_Linear[1,4]
Info_Linear[1,5] = np.mean(I_2_Linear*I_6_Linear)
Info_Linear[5,1] = Info_Linear[1,5]
Info_Linear[1,6] = np.mean(I_2_Linear*I_7_Linear)
Info_Linear[6,1] = Info_Linear[1,6]
Info_Linear[2,3] = np.mean(I_3_Linear*I_4_Linear)
Info_Linear[3,2] = Info_Linear[2,3]
Info_Linear[2,4] = np.mean(I_3_Linear*I_5_Linear)
Info_Linear[4,2] = Info_Linear[2,4]
Info_Linear[2,5] = np.mean(I_3_Linear*I_6_Linear)
Info_Linear[5,2] = Info_Linear[2,5]
Info_Linear[2,6] = np.mean(I_3_Linear*I_7_Linear)
Info_Linear[6,2] = Info_Linear[2,6]
Info_Linear[3,4] = np.mean(I_4_Linear*I_5_Linear)
Info_Linear[4,3] = Info_Linear[3,4]
Info_Linear[3,5] = np.mean(I_4_Linear*I_6_Linear)
Info_Linear[5,3] = Info_Linear[3,5]
Info_Linear[3,6] = np.mean(I_4_Linear*I_7_Linear)
Info_Linear[6,3] = Info_Linear[3,6]
Info_Linear[4,5] = np.mean(I_5_Linear*I_6_Linear)
Info_Linear[5,4] = Info_Linear[4,5]
Info_Linear[4,6] = np.mean(I_5_Linear*I_7_Linear)
Info_Linear[6,4] = Info_Linear[4,6]
Info_Linear[5,6] = np.mean(I_6_Linear*I_7_Linear)
Info_Linear[6,5] = Info_Linear[5,6]

Sigma_Linear = np.linalg.inv(Info_Linear)/len(U_R_train)
dic_L = {"gamma_L": theta_Linear[n_L:(2*n_L)], "sd_L": np.sqrt(np.diag(Sigma_Linear))}
Result_L = pd.DataFrame(dic_L,index=['gamma1','gamma2','gamma3','gamma4','gamma_age','gamma_systolic','gamma_diastolic'])
Result_L.to_csv('Result_L.csv')

dic_g5 = {"g_5": g_other5}
Result_g5 = pd.DataFrame(dic_g5,index=['g1','g2','g3','g4','g5'])
Result_g5.to_csv('Result_g5.csv')

dic_D = {"gamma_deep": theta_Deep[4:8], "sd_deep": np.sqrt(np.diag(Sigma_Deep))}
Result_deep = pd.DataFrame(dic_D,index=['gamma1','gamma2','gamma3','gamma4'])
Result_deep.to_csv('Result_deep.csv')

# # ------------------------------------
# ax_g_diff.scatter(np.arange(X_R_train.shape[0]), Est_Deep['g_train'][:,1]-Est_Linear['g_train'][:,1], s=4)
# fig_g_diff.savefig('fig_g_diff.jpeg', dpi=400, bbox_inches='tight')
# # ------------------------------------
# ax_g_deep.scatter(np.arange(X_R_train.shape[0]), Est_Deep['g_train'][:,1], s=4)
# fig_g_deep.savefig('fig_g_deep.jpeg', dpi=400, bbox_inches='tight')

# ax_g_Cox.scatter(np.arange(X_R_train.shape[0]), Est_Linear['g_train'][:,1], s=4)
# fig_g_Cox.savefig('fig_g_Cox.jpeg', dpi=400, bbox_inches='tight')

# ax_g_deep_no.scatter(np.arange(X_R_train.shape[0]), Est_Deep['g_train'][:,1], s=4)
# fig_g_deep_no.savefig('fig_g_deep_no.jpeg', dpi=400, bbox_inches='tight')

# ax_g_Cox_no.scatter(np.arange(X_R_train.shape[0]), Est_Linear['g_train'][:,1], s=4)
# fig_g_Cox_no.savefig('fig_g_Cox_no.jpeg', dpi=400, bbox_inches='tight')


#%% =================gamma1,gamma2,gamma3,gamma4=================
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
ax1.set_title(r'$(a)\quad\hat\gamma_1$', fontsize=10) 
ax1.set_xlabel("Fold",fontsize=8) 
ax1.set_ylabel("Estimates of effect",fontsize=8) 
ax1.tick_params(axis='both',labelsize=6) 
ax1.grid(True)

fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
ax2.set_title(r'$(b)\quad\hat\gamma_2$', fontsize=10) 
ax2.set_xlabel("Fold",fontsize=8) 
ax2.set_ylabel("Estimates of effect",fontsize=8) 
ax2.tick_params(axis='both',labelsize=6) 
ax2.grid(True)

fig3 = plt.figure()
ax3 = fig3.add_subplot(1, 1, 1)
ax3.set_title(r'$(c)\quad\hat\gamma_3$', fontsize=10) 
ax3.set_xlabel("Fold",fontsize=8) 
ax3.set_ylabel("Estimates of effect",fontsize=8) 
ax3.tick_params(axis='both',labelsize=6) 
ax3.grid(True)

fig4 = plt.figure()
ax4 = fig4.add_subplot(1, 1, 1)
ax4.set_title(r'$(d)\quad\hat\gamma_4$', fontsize=10) 
ax4.set_xlabel("Fold",fontsize=8) 
ax4.set_ylabel("Estimates of effect",fontsize=8) 
ax4.tick_params(axis='both',labelsize=6) 
ax4.grid(True)



beta_g_D1 = np.zeros((5,n_V1))
gamma_g_D1 = np.zeros((5,n_V1))
beta_g_A1 = np.zeros((5,n_V1))
gamma_g_A1 = np.zeros((5,n_V1))
beta_g_L1 = np.zeros((5,n_V1))
gamma_g_L1 = np.zeros((5,n_V1))


beta_g_D0 = np.zeros((5,n_V0))
gamma_g_D0 = np.zeros((5,n_V0))
beta_g_A0 = np.zeros((5,n_V0))
gamma_g_A0 = np.zeros((5,n_V0))
beta_g_L0 = np.zeros((5,n_V0))
gamma_g_L0 = np.zeros((5,n_V0))

C_D = np.zeros((5, m+p))
C_A = np.zeros((5, m+p))
C_L = np.zeros((5, m+p))

c_n= 2200 
for i in range(5):
    print('i =', i)
    Z_train = np.delete(Z_R_train, np.arange(i*c_n, (i+1)*c_n), axis=0)
    X_train = np.delete(X_R_train, np.arange(i*c_n, (i+1)*c_n), axis=0)
    U_train = np.delete(U_R_train, np.arange(i*c_n, (i+1)*c_n), axis=0)
    V_train = np.delete(V_R_train, np.arange(i*c_n, (i+1)*c_n), axis=0)
    De1_train = np.delete(De1_R_train, np.arange(i*c_n, (i+1)*c_n), axis=0)
    De2_train = np.delete(De2_R_train, np.arange(i*c_n, (i+1)*c_n), axis=0)
    De3_train = np.delete(De3_R_train, np.arange(i*c_n, (i+1)*c_n), axis=0)
    n = len(U_train)
    z_d = Z_train.shape[1]
    #%% DGAHM
    Est_D = Est_deep(Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,X_sort1,X_sort0,theta_initial,n_layer,n_node,n_lr,n_epoch,nodevec,m,c_initial)
    # theta_D
    theta_D = Est_D['theta']
    
    Ezg1_D = np.exp(np.dot(Z_train, theta_D[0:z_d]) + Est_D['g_train'][:,0])
    Ezg2_D = np.exp(np.dot(Z_train, theta_D[z_d:(2*z_d)]) + Est_D['g_train'][:,1])
    # compute \Lambda()
    Iu_D = I_U(m, U_train * Ezg1_D, nodevec)
    Iv_D = I_U(m, V_train * Ezg1_D, nodevec)
    Lamb_U_D = np.dot(Iu_D, Est_D['C'])
    Lamb_V_D = np.dot(Iv_D, Est_D['C'])
    f_U_D = Lamb_U_D * Ezg2_D
    f_V_D = Lamb_V_D * Ezg2_D
    # compute \Lambda'()
    Bu_D = B_S2(m, U_train * Ezg1_D, nodevec)
    Bv_D = B_S2(m, V_train * Ezg1_D, nodevec)
    dLamb_U_D = np.matmul(Bu_D, Est_D['C'])
    dLamb_V_D = np.matmul(Bv_D, Est_D['C'])
    # compute the least favorable direction for gamma1
    abc_gamma_D1 = LFD_gamma(Z_train[:,0],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Est_D['g_train'],Est_D['theta'],Est_D['C'],m,nodevec,n_layer,n_node=30,n_lr=1e-4,n_epoch=200)
    
    L_gamma_D1 = De1_train * Z_train[:,0] * f_U_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * Z_train[:,0] * (f_V_D * np.exp(-f_V_D)- f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * Z_train[:,0] * f_V_D
    
    L_g1_gamma_D1 = (De1_train * U_train * dLamb_U_D * Ezg1_D * Ezg2_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * Ezg1_D * Ezg2_D * (V_train * dLamb_V_D * np.exp(-f_V_D) - U_train * dLamb_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * V_train * dLamb_V_D * Ezg1_D * Ezg2_D) * abc_gamma_D1[:,0]
    
    L_g2_gamma_D1 = (De1_train * f_U_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * (f_V_D * np.exp(-f_V_D)- f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * f_V_D) * abc_gamma_D1[:,1] 
    
    L_lambda_gamma_D1 = De1_train * f_U_D * np.exp(-f_U_D) * abc_gamma_D1[:,2] / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * (abc_gamma_D1[:,3] * f_V_D * np.exp(-f_V_D)- abc_gamma_D1[:,2] * f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * f_V_D * abc_gamma_D1[:,3]
    
    I_1_D = L_gamma_D1 - L_g1_gamma_D1 - L_g2_gamma_D1 - L_lambda_gamma_D1
    
    # compute the least favorable direction for gamma2
    abc_gamma_D2 = LFD_gamma(Z_train[:,1],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Est_D['g_train'],Est_D['theta'],Est_D['C'],m,nodevec,n_layer,n_node=30,n_lr=1e-4,n_epoch=200)
    
    L_gamma_D2 = De1_train * Z_train[:,1] * f_U_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * Z_train[:,1] * (f_V_D * np.exp(-f_V_D)- f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * Z_train[:,1] * f_V_D
    
    L_g1_gamma_D2 = (De1_train * U_train * dLamb_U_D * Ezg1_D * Ezg2_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * Ezg1_D * Ezg2_D * (V_train * dLamb_V_D * np.exp(-f_V_D) - U_train * dLamb_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * V_train * dLamb_V_D * Ezg1_D * Ezg2_D) * abc_gamma_D2[:,0]
    
    L_g2_gamma_D2 = (De1_train * f_U_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * (f_V_D * np.exp(-f_V_D)- f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * f_V_D) * abc_gamma_D2[:,1] 
    
    L_lambda_gamma_D2 = De1_train * f_U_D * np.exp(-f_U_D) * abc_gamma_D2[:,2] / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * (abc_gamma_D2[:,3] * f_V_D * np.exp(-f_V_D)- abc_gamma_D2[:,2] * f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * f_V_D * abc_gamma_D2[:,3]
    
    I_2_D = L_gamma_D2 - L_g1_gamma_D2 - L_g2_gamma_D2 - L_lambda_gamma_D2
    
    # compute the least favorable direction for gamma3
    abc_gamma_D3 = LFD_gamma(Z_train[:,2],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Est_D['g_train'],Est_D['theta'],Est_D['C'],m,nodevec,n_layer,n_node=30,n_lr=1e-4,n_epoch=200)
    
    L_gamma_D3 = De1_train * Z_train[:,2] * f_U_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * Z_train[:,2] * (f_V_D * np.exp(-f_V_D)- f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * Z_train[:,2] * f_V_D
    
    L_g1_gamma_D3 = (De1_train * U_train * dLamb_U_D * Ezg1_D * Ezg2_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * Ezg1_D * Ezg2_D * (V_train * dLamb_V_D * np.exp(-f_V_D) - U_train * dLamb_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * V_train * dLamb_V_D * Ezg1_D * Ezg2_D) * abc_gamma_D3[:,0]
    
    L_g2_gamma_D3 = (De1_train * f_U_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * (f_V_D * np.exp(-f_V_D)- f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * f_V_D) * abc_gamma_D3[:,1] 
    
    L_lambda_gamma_D3 = De1_train * f_U_D * np.exp(-f_U_D) * abc_gamma_D3[:,2] / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * (abc_gamma_D3[:,3] * f_V_D * np.exp(-f_V_D)- abc_gamma_D3[:,2] * f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * f_V_D * abc_gamma_D3[:,3]
    
    I_3_D = L_gamma_D3 - L_g1_gamma_D3 - L_g2_gamma_D3 - L_lambda_gamma_D3
    
    # compute the least favorable direction for gamma4
    abc_gamma_D4 = LFD_gamma(Z_train[:,3],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Est_D['g_train'],Est_D['theta'],Est_D['C'],m,nodevec,n_layer,n_node=25,n_lr=1e-4,n_epoch=200)
    
    L_gamma_D4 = De1_train * Z_train[:,3] * f_U_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * Z_train[:,3] * (f_V_D * np.exp(-f_V_D)- f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * Z_train[:,3] * f_V_D
    
    L_g1_gamma_D4 = (De1_train * U_train * dLamb_U_D * Ezg1_D * Ezg2_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * Ezg1_D * Ezg2_D * (V_train * dLamb_V_D * np.exp(-f_V_D) - U_train * dLamb_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * V_train * dLamb_V_D * Ezg1_D * Ezg2_D) * abc_gamma_D4[:,0]
    
    L_g2_gamma_D4 = (De1_train * f_U_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * (f_V_D * np.exp(-f_V_D)- f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * f_V_D) * abc_gamma_D4[:,1] 
    
    L_lambda_gamma_D4 = De1_train * f_U_D * np.exp(-f_U_D) * abc_gamma_D4[:,2] / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * (abc_gamma_D4[:,3] * f_V_D * np.exp(-f_V_D)- abc_gamma_D4[:,2] * f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * f_V_D * abc_gamma_D4[:,3]
    
    I_4_D = L_gamma_D4 - L_g1_gamma_D4 - L_g2_gamma_D4 - L_lambda_gamma_D4
    
    
    Info_D = np.zeros((4,4))
    Info_D[0,0] = np.mean(I_1_D**2)
    Info_D[1,1] = np.mean(I_2_D**2)
    Info_D[2,2] = np.mean(I_3_D**2)
    Info_D[3,3] = np.mean(I_4_D**2)
    Info_D[0,1] = np.mean(I_1_D*I_2_D)
    Info_D[1,0] = Info_D[0,1]
    Info_D[0,2] = np.mean(I_1_D*I_3_D)
    Info_D[2,0] = Info_D[0,2]
    Info_D[0,3] = np.mean(I_1_D*I_4_D)
    Info_D[3,0] = Info_D[0,3]
    Info_D[1,2] = np.mean(I_2_D*I_3_D)
    Info_D[2,1] = Info_D[1,2]
    Info_D[1,3] = np.mean(I_2_D*I_4_D)
    Info_D[3,1] = Info_D[1,3]
    Info_D[2,3] = np.mean(I_3_D*I_4_D)
    Info_D[3,2] = Info_D[2,3]
    Sigma_D = np.linalg.inv(Info_D)/n
    sd1_D = np.sqrt(Sigma_D[0,0])
    sd2_D = np.sqrt(Sigma_D[1,1])
    sd3_D = np.sqrt(Sigma_D[2,2])
    sd4_D = np.sqrt(Sigma_D[3,3])
    # ----------PDLCM------------
    y_min1 = theta_D[4] - 1.96*sd1_D
    y_max1 = theta_D[4] + 1.96*sd1_D
    
    ax1.plot(i+1-0.1, theta_D[4], marker='s', markersize=4, ls='-', label='DGAHM', color='blue')
    if (i == 0):
        ax1.legend(loc='best', fontsize=6)
    
    ax1.plot((i+1-0.1)*np.ones(2), np.array([y_min1, y_max1]), color='blue', marker='_', ls='-')
    # ------------PDLCM-----------
    y_min2 = theta_D[5] - 1.96*sd2_D
    y_max2 = theta_D[5] + 1.96*sd2_D
    
    ax2.plot(i+1-0.1, theta_D[5], marker='s', markersize=4, ls='-',label='DGAHM', color='blue')
    if (i == 0):
        ax2.legend(loc='best', fontsize=6)
    
    ax2.plot((i+1-0.1)*np.ones(2), np.array([y_min2, y_max2]), color='blue', marker='_', ls='-')
    # ------------PDLCM-----------
    y_min3 = theta_D[6] - 1.96*sd3_D
    y_max3 = theta_D[6] + 1.96*sd3_D
    
    ax3.plot(i+1-0.1, theta_D[6], marker='s', markersize=4, ls='-', label='DGAHM', color='blue')
    if (i == 0):
        ax3.legend(loc='best', fontsize=6)
    
    ax3.plot((i+1-0.1)*np.ones(2), np.array([y_min3, y_max3]), color='blue', marker='_', ls='-')
    # ------------PDLCM-----------
    y_min4 = theta_D[7] - 1.96*sd4_D
    y_max4 = theta_D[7] + 1.96*sd4_D
    
    ax4.plot(i+1-0.1, theta_D[7], marker='s', markersize=4, ls='-', label='DGAHM', color='blue')
    if (i == 0):
        ax4.legend(loc='best', fontsize=6)
    
    ax4.plot((i+1-0.1)*np.ones(2), np.array([y_min4, y_max4]), color='blue', marker='_', ls='-')
    
    
    beta_g_D1[i] = np.dot(Z_sort1, theta_D[0:z_d]) + Est_D['g_test1'][:,0]
    gamma_g_D1[i] = np.dot(Z_sort1, theta_D[z_d:(2*z_d)]) + Est_D['g_test1'][:,1]
    beta_g_D0[i] = np.dot(Z_sort0, theta_D[0:z_d]) + Est_D['g_test0'][:,0]
    gamma_g_D0[i] = np.dot(Z_sort0, theta_D[z_d:(2*z_d)]) + Est_D['g_test0'][:,1]
    C_D[i] = Est_D['C']
    

    #%% BS-GAHM
    Est_D = Est_deep(Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,X_sort1,X_sort0,theta_initial,n_layer,n_node,n_lr,n_epoch,nodevec,m,c_initial)
    # theta_D
    theta_D = Est_D['theta']
    
    Ezg1_D = np.exp(np.dot(Z_train, theta_D[0:z_d]) + Est_D['g_train'][:,0])
    Ezg2_D = np.exp(np.dot(Z_train, theta_D[z_d:(2*z_d)]) + Est_D['g_train'][:,1])
    # compute \Lambda()
    Iu_D = I_U(m, U_train * Ezg1_D, nodevec)
    Iv_D = I_U(m, V_train * Ezg1_D, nodevec)
    Lamb_U_D = np.dot(Iu_D, Est_D['C'])
    Lamb_V_D = np.dot(Iv_D, Est_D['C'])
    f_U_D = Lamb_U_D * Ezg2_D
    f_V_D = Lamb_V_D * Ezg2_D
    # compute \Lambda'()
    Bu_D = B_S2(m, U_train * Ezg1_D, nodevec)
    Bv_D = B_S2(m, V_train * Ezg1_D, nodevec)
    dLamb_U_D = np.matmul(Bu_D, Est_D['C'])
    dLamb_V_D = np.matmul(Bv_D, Est_D['C'])
    # compute the least favorable direction for gamma1
    abc_gamma_D1 = LFD_gamma(Z_train[:,0],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Est_D['g_train'],Est_D['theta'],Est_D['C'],m,nodevec,n_layer,n_node=30,n_lr=1e-4,n_epoch=200)
    
    L_gamma_D1 = De1_train * Z_train[:,0] * f_U_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * Z_train[:,0] * (f_V_D * np.exp(-f_V_D)- f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * Z_train[:,0] * f_V_D
    
    L_g1_gamma_D1 = (De1_train * U_train * dLamb_U_D * Ezg1_D * Ezg2_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * Ezg1_D * Ezg2_D * (V_train * dLamb_V_D * np.exp(-f_V_D) - U_train * dLamb_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * V_train * dLamb_V_D * Ezg1_D * Ezg2_D) * abc_gamma_D1[:,0]
    
    L_g2_gamma_D1 = (De1_train * f_U_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * (f_V_D * np.exp(-f_V_D)- f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * f_V_D) * abc_gamma_D1[:,1] 
    
    L_lambda_gamma_D1 = De1_train * f_U_D * np.exp(-f_U_D) * abc_gamma_D1[:,2] / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * (abc_gamma_D1[:,3] * f_V_D * np.exp(-f_V_D)- abc_gamma_D1[:,2] * f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * f_V_D * abc_gamma_D1[:,3]
    
    I_1_D = L_gamma_D1 - L_g1_gamma_D1 - L_g2_gamma_D1 - L_lambda_gamma_D1
    
    # compute the least favorable direction for gamma2
    abc_gamma_D2 = LFD_gamma(Z_train[:,1],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Est_D['g_train'],Est_D['theta'],Est_D['C'],m,nodevec,n_layer,n_node=30,n_lr=1e-4,n_epoch=200)
    
    L_gamma_D2 = De1_train * Z_train[:,1] * f_U_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * Z_train[:,1] * (f_V_D * np.exp(-f_V_D)- f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * Z_train[:,1] * f_V_D
    
    L_g1_gamma_D2 = (De1_train * U_train * dLamb_U_D * Ezg1_D * Ezg2_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * Ezg1_D * Ezg2_D * (V_train * dLamb_V_D * np.exp(-f_V_D) - U_train * dLamb_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * V_train * dLamb_V_D * Ezg1_D * Ezg2_D) * abc_gamma_D2[:,0]
    
    L_g2_gamma_D2 = (De1_train * f_U_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * (f_V_D * np.exp(-f_V_D)- f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * f_V_D) * abc_gamma_D2[:,1] 
    
    L_lambda_gamma_D2 = De1_train * f_U_D * np.exp(-f_U_D) * abc_gamma_D2[:,2] / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * (abc_gamma_D2[:,3] * f_V_D * np.exp(-f_V_D)- abc_gamma_D2[:,2] * f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * f_V_D * abc_gamma_D2[:,3]
    
    I_2_D = L_gamma_D2 - L_g1_gamma_D2 - L_g2_gamma_D2 - L_lambda_gamma_D2
    
    # compute the least favorable direction for gamma3
    abc_gamma_D3 = LFD_gamma(Z_train[:,2],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Est_D['g_train'],Est_D['theta'],Est_D['C'],m,nodevec,n_layer,n_node=30,n_lr=1e-4,n_epoch=200)
    
    L_gamma_D3 = De1_train * Z_train[:,2] * f_U_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * Z_train[:,2] * (f_V_D * np.exp(-f_V_D)- f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * Z_train[:,2] * f_V_D
    
    L_g1_gamma_D3 = (De1_train * U_train * dLamb_U_D * Ezg1_D * Ezg2_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * Ezg1_D * Ezg2_D * (V_train * dLamb_V_D * np.exp(-f_V_D) - U_train * dLamb_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * V_train * dLamb_V_D * Ezg1_D * Ezg2_D) * abc_gamma_D3[:,0]
    
    L_g2_gamma_D3 = (De1_train * f_U_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * (f_V_D * np.exp(-f_V_D)- f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * f_V_D) * abc_gamma_D3[:,1] 
    
    L_lambda_gamma_D3 = De1_train * f_U_D * np.exp(-f_U_D) * abc_gamma_D3[:,2] / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * (abc_gamma_D3[:,3] * f_V_D * np.exp(-f_V_D)- abc_gamma_D3[:,2] * f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * f_V_D * abc_gamma_D3[:,3]
    
    I_3_D = L_gamma_D3 - L_g1_gamma_D3 - L_g2_gamma_D3 - L_lambda_gamma_D3
    
    # compute the least favorable direction for gamma4
    abc_gamma_D4 = LFD_gamma(Z_train[:,3],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Est_D['g_train'],Est_D['theta'],Est_D['C'],m,nodevec,n_layer,n_node=25,n_lr=1e-4,n_epoch=200)
    
    L_gamma_D4 = De1_train * Z_train[:,3] * f_U_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * Z_train[:,3] * (f_V_D * np.exp(-f_V_D)- f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * Z_train[:,3] * f_V_D
    
    L_g1_gamma_D4 = (De1_train * U_train * dLamb_U_D * Ezg1_D * Ezg2_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * Ezg1_D * Ezg2_D * (V_train * dLamb_V_D * np.exp(-f_V_D) - U_train * dLamb_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * V_train * dLamb_V_D * Ezg1_D * Ezg2_D) * abc_gamma_D4[:,0]
    
    L_g2_gamma_D4 = (De1_train * f_U_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * (f_V_D * np.exp(-f_V_D)- f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * f_V_D) * abc_gamma_D4[:,1] 
    
    L_lambda_gamma_D4 = De1_train * f_U_D * np.exp(-f_U_D) * abc_gamma_D4[:,2] / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * (abc_gamma_D4[:,3] * f_V_D * np.exp(-f_V_D)- abc_gamma_D4[:,2] * f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * f_V_D * abc_gamma_D4[:,3]
    
    I_4_D = L_gamma_D4 - L_g1_gamma_D4 - L_g2_gamma_D4 - L_lambda_gamma_D4
    
    
    Info_D = np.zeros((4,4))
    Info_D[0,0] = np.mean(I_1_D**2)
    Info_D[1,1] = np.mean(I_2_D**2)
    Info_D[2,2] = np.mean(I_3_D**2)
    Info_D[3,3] = np.mean(I_4_D**2)
    Info_D[0,1] = np.mean(I_1_D*I_2_D)
    Info_D[1,0] = Info_D[0,1]
    Info_D[0,2] = np.mean(I_1_D*I_3_D)
    Info_D[2,0] = Info_D[0,2]
    Info_D[0,3] = np.mean(I_1_D*I_4_D)
    Info_D[3,0] = Info_D[0,3]
    Info_D[1,2] = np.mean(I_2_D*I_3_D)
    Info_D[2,1] = Info_D[1,2]
    Info_D[1,3] = np.mean(I_2_D*I_4_D)
    Info_D[3,1] = Info_D[1,3]
    Info_D[2,3] = np.mean(I_3_D*I_4_D)
    Info_D[3,2] = Info_D[2,3]
    Sigma_D = np.linalg.inv(Info_D)/n
    sd1_D = np.sqrt(Sigma_D[0,0])
    sd2_D = np.sqrt(Sigma_D[1,1])
    sd3_D = np.sqrt(Sigma_D[2,2])
    sd4_D = np.sqrt(Sigma_D[3,3])
    # ----------PDLCM------------
    y_min1 = theta_D[4] - 1.96*sd1_D
    y_max1 = theta_D[4] + 1.96*sd1_D
    
    ax1.plot(i+1-0.1, theta_D[4], marker='s', markersize=4, ls='-', label='BS-GAHM', color='blue')
    if (i == 0):
        ax1.legend(loc='best', fontsize=6)
    
    ax1.plot((i+1-0.1)*np.ones(2), np.array([y_min1, y_max1]), color='blue', marker='_', ls='-')
    # ------------PDLCM-----------
    y_min2 = theta_D[5] - 1.96*sd2_D
    y_max2 = theta_D[5] + 1.96*sd2_D
    
    ax2.plot(i+1-0.1, theta_D[5], marker='s', markersize=4, ls='-',label='BS-GAHM', color='blue')
    if (i == 0):
        ax2.legend(loc='best', fontsize=6)
    
    ax2.plot((i+1-0.1)*np.ones(2), np.array([y_min2, y_max2]), color='blue', marker='_', ls='-')
    # ------------PDLCM-----------
    y_min3 = theta_D[6] - 1.96*sd3_D
    y_max3 = theta_D[6] + 1.96*sd3_D
    
    ax3.plot(i+1-0.1, theta_D[6], marker='s', markersize=4, ls='-', label='BS-GAHM', color='blue')
    if (i == 0):
        ax3.legend(loc='best', fontsize=6)
    
    ax3.plot((i+1-0.1)*np.ones(2), np.array([y_min3, y_max3]), color='blue', marker='_', ls='-')
    # ------------PDLCM-----------
    y_min4 = theta_D[7] - 1.96*sd4_D
    y_max4 = theta_D[7] + 1.96*sd4_D
    
    ax4.plot(i+1-0.1, theta_D[7], marker='s', markersize=4, ls='-', label='BS-GAHM', color='blue')
    if (i == 0):
        ax4.legend(loc='best', fontsize=6)
    
    ax4.plot((i+1-0.1)*np.ones(2), np.array([y_min4, y_max4]), color='blue', marker='_', ls='-')
    
    
    beta_g_D1[i] = np.dot(Z_sort1, theta_D[0:z_d]) + Est_D['g_test1'][:,0]
    gamma_g_D1[i] = np.dot(Z_sort1, theta_D[z_d:(2*z_d)]) + Est_D['g_test1'][:,1]
    beta_g_D0[i] = np.dot(Z_sort0, theta_D[0:z_d]) + Est_D['g_test0'][:,0]
    gamma_g_D0[i] = np.dot(Z_sort0, theta_D[z_d:(2*z_d)]) + Est_D['g_test0'][:,1]
    C_D[i] = Est_D['C']


    #%% GHM
    Est_L = Est_linear(Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,X_sort1,X_sort0,theta_initial,nodevec,m,c_initial)
    
    theta_L = Est_L['theta']
    
    Ezg1_L = np.exp(np.dot(Z_train, theta_L[0:z_d]) + Est_L['g_train'][:,0])
    Ezg2_L = np.exp(np.dot(Z_train, theta_L[z_d:(2*z_d)]) + Est_L['g_train'][:,1])
    # compute \Lambda()
    Iu_L = I_U(m, U_train * Ezg1_L, nodevec)
    Iv_L = I_U(m, V_train * Ezg1_L, nodevec)
    Lamb_U_L = np.dot(Iu_L, Est_L['C'])
    Lamb_V_L = np.dot(Iv_L, Est_L['C'])
    f_U_L = Lamb_U_L * Ezg2_L
    f_V_L = Lamb_V_L * Ezg2_L
    # compute \Lambda'()
    Bu_L = B_S2(m, U_train * Ezg1_L, nodevec)
    Bv_L = B_S2(m, V_train * Ezg1_L, nodevec)
    dLamb_U_L = np.matmul(Bu_L, Est_L['C'])
    dLamb_V_L = np.matmul(Bv_L, Est_L['C'])
    # compute the least favorable direction for gamma1
    abc_gamma_L1 = LFD_gamma(Z_train[:,0],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Est_L['g_train'],Est_L['theta'],Est_L['C'],m,nodevec,n_layer,n_node=30,n_lr=1e-4,n_epoch=200)
    
    L_gamma_L1 = De1_train * Z_train[:,0] * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * Z_train[:,0] * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * Z_train[:,0] * f_V_L
    
    L_g1_gamma_L1 = (De1_train * U_train * dLamb_U_L * Ezg1_L * Ezg2_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * Ezg1_L * Ezg2_L * (V_train * dLamb_V_L * np.exp(-f_V_L) - U_train * dLamb_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * V_train * dLamb_V_L * Ezg1_L * Ezg2_L) * abc_gamma_L1[:,0]
    
    L_g2_gamma_L1 = (De1_train * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * f_V_L) * abc_gamma_L1[:,1] 
    
    L_lambda_gamma_L1 = De1_train * f_U_L * np.exp(-f_U_L) * abc_gamma_L1[:,2] / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * (abc_gamma_L1[:,3] * f_V_L * np.exp(-f_V_L)- abc_gamma_L1[:,2] * f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * f_V_L * abc_gamma_L1[:,3]
    
    I_1_L = L_gamma_L1 - L_g1_gamma_L1 - L_g2_gamma_L1 - L_lambda_gamma_L1
    
    # compute the least favorable direction for gamma2
    abc_gamma_L2 = LFD_gamma(Z_train[:,1],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Est_L['g_train'],Est_L['theta'],Est_L['C'],m,nodevec,n_layer,n_node=30,n_lr=1e-4,n_epoch=200)
    
    L_gamma_L2 = De1_train * Z_train[:,1] * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * Z_train[:,1] * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * Z_train[:,1] * f_V_L
    
    L_g1_gamma_L2 = (De1_train * U_train * dLamb_U_L * Ezg1_L * Ezg2_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * Ezg1_L * Ezg2_L * (V_train * dLamb_V_L * np.exp(-f_V_L) - U_train * dLamb_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * V_train * dLamb_V_L * Ezg1_L * Ezg2_L) * abc_gamma_L2[:,0]
    
    L_g2_gamma_L2 = (De1_train * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * f_V_L) * abc_gamma_L2[:,1] 
    
    L_lambda_gamma_L2 = De1_train * f_U_L * np.exp(-f_U_L) * abc_gamma_L2[:,2] / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * (abc_gamma_L2[:,3] * f_V_L * np.exp(-f_V_L)- abc_gamma_L2[:,2] * f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * f_V_L * abc_gamma_L2[:,3]
    
    I_2_L = L_gamma_L2 - L_g1_gamma_L2 - L_g2_gamma_L2 - L_lambda_gamma_L2
    
    # compute the least favorable direction for gamma3
    abc_gamma_L3 = LFD_gamma(Z_train[:,2],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Est_L['g_train'],Est_L['theta'],Est_L['C'],m,nodevec,n_layer,n_node=30,n_lr=1e-4,n_epoch=200)
    
    L_gamma_L3 = De1_train * Z_train[:,2] * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * Z_train[:,2] * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * Z_train[:,2] * f_V_L
    
    L_g1_gamma_L3 = (De1_train * U_train * dLamb_U_L * Ezg1_L * Ezg2_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * Ezg1_L * Ezg2_L * (V_train * dLamb_V_L * np.exp(-f_V_L) - U_train * dLamb_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * V_train * dLamb_V_L * Ezg1_L * Ezg2_L) * abc_gamma_L3[:,0]
    
    L_g2_gamma_L3 = (De1_train * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * f_V_L) * abc_gamma_L3[:,1] 
    
    L_lambda_gamma_L3 = De1_train * f_U_L * np.exp(-f_U_L) * abc_gamma_L3[:,2] / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * (abc_gamma_L3[:,3] * f_V_L * np.exp(-f_V_L)- abc_gamma_L3[:,2] * f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * f_V_L * abc_gamma_L3[:,3]
    
    I_3_L = L_gamma_L3 - L_g1_gamma_L3 - L_g2_gamma_L3 - L_lambda_gamma_L3
    
    # compute the least favorable direction for gamma4
    abc_gamma_L4 = LFD_gamma(Z_train[:,3],Z_train,X_train,U_train,V_train,De1_train,De2_train,De3_train,Est_L['g_train'],Est_L['theta'],Est_L['C'],m,nodevec,n_layer,n_node=30,n_lr=1e-4,n_epoch=200)
    
    L_gamma_L4 = De1_train * Z_train[:,3] * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * Z_train[:,3] * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * Z_train[:,3] * f_V_L
    
    L_g1_gamma_L4 = (De1_train * U_train * dLamb_U_L * Ezg1_L * Ezg2_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * Ezg1_L * Ezg2_L * (V_train * dLamb_V_L * np.exp(-f_V_L) - U_train * dLamb_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * V_train * dLamb_V_L * Ezg1_L * Ezg2_L) * abc_gamma_L4[:,0]
    
    L_g2_gamma_L4 = (De1_train * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * f_V_L) * abc_gamma_L4[:,1] 
    
    L_lambda_gamma_L4 = De1_train * f_U_L * np.exp(-f_U_L) * abc_gamma_L4[:,2] / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * (abc_gamma_L4[:,3] * f_V_L * np.exp(-f_V_L)- abc_gamma_L4[:,2] * f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * f_V_L * abc_gamma_L4[:,3]
    
    I_4_L = L_gamma_L4 - L_g1_gamma_L4 - L_g2_gamma_L4 - L_lambda_gamma_L4
    
    
    Info_L = np.zeros((4,4))
    Info_L[0,0] = np.mean(I_1_L**2)
    Info_L[1,1] = np.mean(I_2_L**2)
    Info_L[2,2] = np.mean(I_3_L**2)
    Info_L[3,3] = np.mean(I_4_L**2)
    Info_L[0,1] = np.mean(I_1_L*I_2_L)
    Info_L[1,0] = Info_L[0,1]
    Info_L[0,2] = np.mean(I_1_L*I_3_L)
    Info_L[2,0] = Info_L[0,2]
    Info_L[0,3] = np.mean(I_1_L*I_4_L)
    Info_L[3,0] = Info_L[0,3]
    Info_L[1,2] = np.mean(I_2_L*I_3_L)
    Info_L[2,1] = Info_L[1,2]
    Info_L[1,3] = np.mean(I_2_L*I_4_L)
    Info_L[3,1] = Info_L[1,3]
    Info_L[2,3] = np.mean(I_3_L*I_4_L)
    Info_L[3,2] = Info_L[2,3]
    Sigma_L = np.linalg.inv(Info_L)/n
    sd1_L = np.sqrt(Sigma_L[0,0])
    sd2_L = np.sqrt(Sigma_L[1,1])
    sd3_L = np.sqrt(Sigma_L[2,2])
    sd4_L = np.sqrt(Sigma_L[3,3])
    # ----------GHM-----------
    y_min1 = theta_L[4] - 1.96*sd1_L
    y_max1 = theta_L[4] + 1.96*sd1_L
    
    ax1.plot(i+1+0.1, theta_L[4],  marker='o', markersize=4, ls='-', label='GHM', color='orange')
    if (i == 0):
        ax1.legend(loc='best', fontsize=6)
    
    ax1.plot((i+1+0.1)*np.ones(2), np.array([y_min1, y_max1]), color='orange', marker='_', ls='-')
    # ------------GHM-----------
    y_min2 = theta_L[5] - 1.96*sd2_L
    y_max2 = theta_L[5] + 1.96*sd2_L
    
    ax2.plot(i+1+0.1, theta_L[5],  marker='o', markersize=4, ls='-',label='GHM', color='orange')
    if (i == 0):
        ax2.legend(loc='best', fontsize=6)
    
    ax2.plot((i+1+0.1)*np.ones(2), np.array([y_min2, y_max2]), color='orange', marker='_', ls='-')
    # ------------GHM-----------
    y_min3 = theta_L[6] - 1.96*sd3_L
    y_max3 = theta_L[6] + 1.96*sd3_L
    
    ax3.plot(i+1+0.1, theta_L[6],  marker='o', markersize=4, ls='-', label='GHM', color='orange')
    if (i == 0):
        ax3.legend(loc='best', fontsize=6)
    
    ax3.plot((i+1+0.1)*np.ones(2), np.array([y_min3, y_max3]), color='orange', marker='_', ls='-')
    # ------------GHM-----------
    y_min4 = theta_L[7] - 1.96*sd4_L
    y_max4 = theta_L[7] + 1.96*sd4_L
    
    ax4.plot(i+1+0.1, theta_L[7],  marker='o', markersize=4, ls='-', label='GHM', color='orange')
    if (i == 0):
        ax4.legend(loc='best', fontsize=6)
    
    ax4.plot((i+1+0.1)*np.ones(2), np.array([y_min4, y_max4]), color='orange', marker='_', ls='-')
    
    beta_g_L1[i] = np.dot(Z_sort1, theta_L[0:z_d]) + Est_L['g_test1'][:,0]
    gamma_g_L1[i] = np.dot(Z_sort1, theta_L[z_d:(2*z_d)]) + Est_L['g_test1'][:,1]
    beta_g_L0[i] = np.dot(Z_sort0, theta_L[0:z_d]) + Est_L['g_test0'][:,0]
    gamma_g_L0[i] = np.dot(Z_sort0, theta_L[z_d:(2*z_d)]) + Est_L['g_test0'][:,1]
    C_L[i] = Est_L['C']

# ===============================================
fig1.savefig('fig1.jpeg', dpi=400, bbox_inches='tight')
fig2.savefig('fig2.jpeg', dpi=400, bbox_inches='tight')
fig3.savefig('fig3.jpeg', dpi=400, bbox_inches='tight')
fig4.savefig('fig4.jpeg', dpi=400, bbox_inches='tight')
# =======================================================


Beta_g_D1 = np.mean(beta_g_D1,axis=0) 
Gamma_g_D1 = np.mean(gamma_g_D1,axis=0) 
Beta_g_D0 = np.mean(beta_g_D0,axis=0) 
Gamma_g_D0 = np.mean(gamma_g_D0,axis=0) 
C_D = np.mean(np.array(C_D), axis=0) 
Lamd_U_D1 = I_S(m,C_D,U_sort1*Beta_g_D1,nodevec) 
Lamd_V_D1 = I_S(m,C_D,V_sort1*Beta_g_D1,nodevec) 
Lamd_U_D0 = I_S(m,C_D,U_sort0*Beta_g_D0,nodevec) 
Lamd_V_D0 = I_S(m,C_D,V_sort0*Beta_g_D0,nodevec) 


Beta_g_L1 = np.mean(beta_g_L1,axis=0) 
Gamma_g_L1 = np.mean(gamma_g_L1,axis=0) 
Beta_g_L0 = np.mean(beta_g_L0,axis=0) 
Gamma_g_L0 = np.mean(gamma_g_L0,axis=0) 
C_L = np.mean(np.array(C_L), axis=0) 
Lamd_U_L1 = I_S(m,C_L,U_sort1*Beta_g_L1,nodevec) 
Lamd_V_L1 = I_S(m,C_L,V_sort1*Beta_g_L1,nodevec) 
Lamd_U_L0 = I_S(m,C_L,U_sort0*Beta_g_L0,nodevec) 
Lamd_V_L0 = I_S(m,C_L,V_sort0*Beta_g_L0,nodevec) 

#%% Prediction of survival function for 6 subjects
# Calculate and draw three graphs with delta3 = 1
for k in range(3):
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(1, 1, 1)
    ax5.set_xlabel("t",fontsize=8)      
    ax5.set_ylabel(r'$\hat{S}(t)$',fontsize=8)
    ax5.tick_params(axis='both',labelsize=6)
    # Shift the position, set the origin to intersect
    ax5.xaxis.set_ticks_position('bottom')
    ax5.spines['bottom'].set_position(('data',0))
    ax5.yaxis.set_ticks_position('left')
    ax5.spines['left'].set_position(('data',0))
    ax5.grid(True)
    # Calculate S(t)
    Beta_g_D_1_k = np.exp(Beta_g_D1[round(n_V1*0.25*(k+1))])
    Gamma_g_D_1_k = Gamma_g_D1[round(n_V1*0.25*(k+1))]
    St_D1 = np.exp(-I_S(m,C_D,V1_value*Beta_g_D_1_k,nodevec) * np.exp(Gamma_g_D_1_k))
    
    Beta_g_L_1_k = np.exp(Beta_g_L1[round(n_V1*0.25*(k+1))])
    Gamma_g_L_1_k = Gamma_g_L1[round(n_V1*0.25*(k+1))]
    St_L1 = np.exp(-I_S(m,C_L,V1_value*Beta_g_L_1_k,nodevec) * np.exp(Gamma_g_L_1_k))
    
    # drawing 
    ax5.plot(V1_value, St_D1, color='blue', linestyle=':')
    ax5.plot(V1_value, St_L1, color='orange', linestyle='--')
    ax5.plot(V1_value, 0.5*np.ones(len(V1_value)), color='red', linestyle='-')
    # ax5.legend(loc='best', fontsize=6)
    # save figures
    if (k==0):
        ax5.plot(V1_025, np.exp(-I_S(m,C_D,np.array([V1_025*Beta_g_D_1_k]),nodevec) * np.exp(Gamma_g_D_1_k)), label='DGAHM', marker='s', markersize=4, ls='-', color='blue')
        ax5.plot(V1_025, np.exp(-I_S(m,C_L,np.array([V1_025*Beta_g_L_1_k]),nodevec) * np.exp(Gamma_g_L_1_k)), label='GHM', marker='o', markersize=4, ls='--', color='orange')
        ax5.plot(np.array([V1_025,V1_025]), np.array([0,np.max([np.exp(-I_S(m,C_D,np.array([V1_025*Beta_g_D_1_k]),nodevec) * np.exp(Gamma_g_D_1_k)), np.exp(-I_S(m,C_L,np.array([V1_025*Beta_g_L_1_k]),nodevec) * np.exp(Gamma_g_L_1_k))])], dtype='float32'), color='k', linestyle='--')
        ax5.legend(loc='best', fontsize=6)
        ax5.set_title(r'$\Delta_3=1, 25^{\rm{th}}$', fontsize=10) 
        fig5.savefig('fig1_25.jpeg', dpi=400, bbox_inches='tight')
    elif (k==1):
        ax5.plot(V1_050, np.exp(-I_S(m,C_D,np.array([V1_050*Beta_g_D_1_k]),nodevec) * np.exp(Gamma_g_D_1_k)), label='DGAHM', marker='s', markersize=4, ls='-', color='blue')
        ax5.plot(V1_050, np.exp(-I_S(m,C_L,np.array([V1_050*Beta_g_L_1_k]),nodevec) * np.exp(Gamma_g_L_1_k)), label='GHM', marker='o', markersize=4, ls='--', color='orange')
        ax5.plot(np.array([V1_050,V1_050]), np.array([0,np.max([np.exp(-I_S(m,C_D,np.array([V1_050*Beta_g_D_1_k]),nodevec) * np.exp(Gamma_g_D_1_k)), np.exp(-I_S(m,C_L,np.array([V1_050*Beta_g_L_1_k]),nodevec) * np.exp(Gamma_g_L_1_k))])], dtype='float32'), color='k', linestyle='--')
        ax5.legend(loc='best', fontsize=6)
        ax5.set_title(r'$\Delta_3=1, 50^{\rm{th}}$', fontsize=10) 
        fig5.savefig('fig1_50.jpeg', dpi=400, bbox_inches='tight')
    else:
        ax5.plot(V1_075, np.exp(-I_S(m,C_D,np.array([V1_075*Beta_g_D_1_k]),nodevec) * np.exp(Gamma_g_D_1_k)), label='DGAHM', marker='s', markersize=4, ls='-', color='blue')
        ax5.plot(V1_075, np.exp(-I_S(m,C_L,np.array([V1_075*Beta_g_L_1_k]),nodevec) * np.exp(Gamma_g_L_1_k)), label='GHM', marker='o', markersize=4, ls='--', color='orange')
        ax5.plot(np.array([V1_075,V1_075]), np.array([0,np.max([np.exp(-I_S(m,C_D,np.array([V1_075*Beta_g_D_1_k]),nodevec) * np.exp(Gamma_g_D_1_k)), np.exp(-I_S(m,C_L,np.array([V1_075*Beta_g_L_1_k]),nodevec) * np.exp(Gamma_g_L_1_k))])], dtype='float32'), color='k', linestyle='--')
        ax5.legend(loc='best', fontsize=6)
        ax5.set_title(r'$\Delta_3=1, 75^{\rm{th}}$', fontsize=10) 
        fig5.savefig('fig1_75.jpeg', dpi=400, bbox_inches='tight')

# Calculate and draw three figures with delta3=0
for k in range(3):
    fig6 = plt.figure()
    ax6 = fig6.add_subplot(1, 1, 1)
    ax6.set_xlabel("t",fontsize=8)       
    ax6.set_ylabel(r'$\hat{S}(t)$',fontsize=8) 
    ax6.tick_params(axis='both',labelsize=6) 
    # Shift the position, set the origin to intersect
    ax6.xaxis.set_ticks_position('bottom')
    ax6.spines['bottom'].set_position(('data',0))
    ax6.yaxis.set_ticks_position('left')
    ax6.spines['left'].set_position(('data',0))
    ax6.grid(True)
    # Calculate S(t)
    Beta_g_D_0_k = np.exp(Beta_g_D0[round(n_V0*0.25*(k+1))])
    Gamma_g_D_0_k = Gamma_g_D0[round(n_V0*0.25*(k+1))]
    St_D0 = np.exp(-I_S(m,C_D,V0_value*Beta_g_D_0_k,nodevec) * np.exp(Gamma_g_D_0_k))
    
    Beta_g_L_0_k = np.exp(Beta_g_L0[round(n_V0*0.25*(k+1))])
    Gamma_g_L_0_k = Gamma_g_L0[round(n_V0*0.25*(k+1))]
    St_L0 = np.exp(-I_S(m,C_L,V0_value*Beta_g_L_0_k,nodevec) * np.exp(Gamma_g_L_0_k))
    
    # drawing 
    ax6.plot(V0_value, St_D0, color='blue', linestyle=':')
    ax6.plot(V0_value, St_L0, color='orange', linestyle='--')
    ax6.plot(V0_value, 0.5*np.ones(len(V0_value)), color='red', linestyle='-')
    # ax6.legend(loc='best', fontsize=6)
    # save figures
    if (k==0):
        ax6.plot(V0_025, np.exp(-I_S(m,C_D,np.array([V0_025*Beta_g_D_0_k]),nodevec) * np.exp(Gamma_g_D_0_k)), label='DGAHM', marker='s', markersize=4, ls='-', color='blue')
        ax6.plot(V0_025, np.exp(-I_S(m,C_L,np.array([V0_025*Beta_g_L_0_k]),nodevec) * np.exp(Gamma_g_L_0_k)), label='GHM', marker='o', markersize=4, ls='--', color='orange')
        ax6.plot(np.array([V0_025,V0_025]), np.array([0,np.max([np.exp(-I_S(m,C_D,np.array([V0_025*Beta_g_D_0_k]),nodevec) * np.exp(Gamma_g_D_0_k)), np.exp(-I_S(m,C_L,np.array([V0_025*Beta_g_L_0_k]),nodevec) * np.exp(Gamma_g_L_0_k))])], dtype='float32'), color='k', linestyle='--')
        ax6.legend(loc='best', fontsize=6)
        ax6.set_title(r'$\Delta_3=0, 25^{\rm{th}}$', fontsize=10) 
        fig6.savefig('fig0_25.jpeg', dpi=400, bbox_inches='tight')
    elif (k==1):
        ax6.plot(V0_050, np.exp(-I_S(m,C_D,np.array([V0_050*Beta_g_D_0_k]),nodevec) * np.exp(Gamma_g_D_0_k)), label='DGAHM', marker='s', markersize=4, ls='-', color='blue')
        ax6.plot(V0_050, np.exp(-I_S(m,C_L,np.array([V0_050*Beta_g_L_0_k]),nodevec) * np.exp(Gamma_g_L_0_k)), label='GHM', marker='o', markersize=4, ls='--', color='orange')
        ax6.plot(np.array([V0_050,V0_050]), np.array([0,np.max([np.exp(-I_S(m,C_D,np.array([V0_050*Beta_g_D_0_k]),nodevec) * np.exp(Gamma_g_D_0_k)), np.exp(-I_S(m,C_L,np.array([V0_050*Beta_g_L_0_k]),nodevec) * np.exp(Gamma_g_L_0_k))])], dtype='float32'), color='k', linestyle='--')
        ax6.legend(loc='best', fontsize=6)
        ax6.set_title(r'$\Delta_3=0, 50^{\rm{th}}$', fontsize=10) 
        fig6.savefig('fig0_50.jpeg', dpi=400, bbox_inches='tight')
    else:
        ax6.plot(V0_075, np.exp(-I_S(m,C_D,np.array([V0_075*Beta_g_D_0_k]),nodevec) * np.exp(Gamma_g_D_0_k)), label='DGAHM', marker='s', markersize=4, ls='-', color='blue')
        ax6.plot(V0_075, np.exp(-I_S(m,C_L,np.array([V0_075*Beta_g_L_0_k]),nodevec) * np.exp(Gamma_g_L_0_k)), label='GHM', marker='o', markersize=4, ls='--', color='orange')
        ax6.plot(np.array([V0_075,V0_075]), np.array([0,np.max([np.exp(-I_S(m,C_D,np.array([V0_075*Beta_g_D_0_k]),nodevec) * np.exp(Gamma_g_D_0_k)), np.exp(-I_S(m,C_L,np.array([V0_075*Beta_g_L_0_k]),nodevec) * np.exp(Gamma_g_L_0_k))])], dtype='float32'), color='k', linestyle='--')
        ax6.legend(loc='best', fontsize=6)
        ax6.set_title(r'$\Delta_3=0, 75^{\rm{th}}$', fontsize=10) 
        fig6.savefig('fig0_75.jpeg', dpi=400, bbox_inches='tight')        

