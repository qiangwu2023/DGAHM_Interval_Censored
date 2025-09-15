#%% ----------- import packages -----------
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from data_generator import generate_case_4
from iteration_deep import Est_deep
from iteration_additive import Est_additive
from iteration_linear import Est_linear
from Least_beta import LFD_beta
from Least_gamma import LFD_gamma
from I_spline import I_U
from I_spline import I_S
from B_spline2 import B_S2
#%% ---------- define seeds -------------
def set_seed(seed):
    np.random.seed(seed)  
    torch.manual_seed(seed) 
#%% ---------- set seed -------------
set_seed(8)  
#%% -----------------------
tau = 5 
p = 3 
Set_n = np.array([500, 1000]) 
corr = 0.5 
n_layer = 3 
Set_node = np.array([50, 50])
n_epoch = 200
Set_lr = np.array([5e-4, 5e-4]) 
theta0 = np.array([2, 1], dtype="float32") 
beta_node_D = np.array([30, 44]) 
beta_lr_D = np.array([3e-4, 2e-4]) 
gamma_node_D = np.array([52, 50])
gamma_lr_D = np.array([3e-5, 3e-5])
beta_node_A = np.array([45, 45]) 
beta_lr_A = np.array([5e-4, 5e-4])
gamma_node_A = np.array([45, 45]) 
gamma_lr_A = np.array([2e-5, 2e-5]) 
beta_node_L = np.array([45, 45]) 
beta_lr_L = np.array([5e-4, 5e-4])
gamma_node_L = np.array([45, 45]) 
gamma_lr_L = np.array([2e-5, 2e-5]) 

B = 200 
#%% test data
test_data = generate_case_4(200, corr, theta0, tau)
X_test = test_data['X']
g1_true = test_data['g1_X']
g2_true = test_data['g2_X']
dim_x = X_test.shape[0]

m = 10 
nodevec = np.array(np.linspace(0, 83, m+2), dtype="float32") 

m0 = 4 # the number of interior knot set of B-spline functions
nodevec0 = np.array(np.linspace(0, 1, m0+2), dtype="float32") # the knot set of B-spline basis functions
#%% X_1,X_2, Z=0,1
X_subject = np.array([[0.1,0.3,0.5,0.7],[0.8,0.5,0.2,0]], dtype='float32')
Z_subject = np.array([0, 1], dtype='float32')
g_1_X_subject_true = (np.sqrt(X_subject[:,0] * X_subject[:,1]) / 2 + np.log(X_subject[:,1] * X_subject[:,2]+ 1) / 3 + np.exp(X_subject[:,3]) / 4) ** 2 / 2 - 0.19
g_2_X_subject_true = (X_subject[:,0] * X_subject[:,1] / 2 + X_subject[:,2] ** 2 * X_subject[:,3] / 3 + np.log(X_subject[:,3] + 1)) ** 2 / 2 - 0.16

t_value = np.array(np.linspace(0, tau, 30), dtype="float32")
Lambda_Z0_X1 = (t_value * np.exp(g_1_X_subject_true[0])) ** (3/4) * np.exp(g_2_X_subject_true[0]) / 6 # Z=0, X=X1
Lambda_Z1_X1 = (t_value * np.exp(theta0[0] + g_1_X_subject_true[0])) ** (3/4)  * np.exp(theta0[1] + g_2_X_subject_true[0]) / 6 # Z=1, X=X1
Lambda_Z0_X2 = (t_value * np.exp(g_1_X_subject_true[1])) ** (3/4) * np.exp(g_2_X_subject_true[1]) / 6 # Z=0, X=X2
Lambda_Z1_X2 = (t_value * np.exp(theta0[0] + g_1_X_subject_true[1])) ** (3/4) * np.exp(theta0[1] + g_2_X_subject_true[1]) / 6 # Z=1, X=X2

#%% ---------------- Main results -----------------------
#%% Survival function
# ------(prediction)---(Z=0,X=X1)------
fig1_Survival = plt.figure()
ax1_1_Survival = fig1_Survival.add_subplot(1, 2, 1)
ax1_1_Survival.set_title("Case 4, n=500, $\Lambda_0(t)=t^{3/4}/6$", fontsize=10) 
ax1_1_Survival.set_xlabel("Time",fontsize=8) 
ax1_1_Survival.set_ylabel("Survival function",fontsize=8) 
ax1_1_Survival.tick_params(axis='both',labelsize=6) 
ax1_1_Survival.plot(t_value, np.exp(-Lambda_Z0_X1), color='k', label='True') 
ax1_1_Survival.legend(loc='upper left', fontsize=6) 

ax1_2_Survival = fig1_Survival.add_subplot(1, 2, 2)
ax1_2_Survival.set_title("Case 4, n=1000, $\Lambda_0(t)=t^{3/4}/6$", fontsize=10) 
ax1_2_Survival.set_xlabel("Time",fontsize=8) 
ax1_2_Survival.set_ylabel("Survival function",fontsize=8) 
ax1_2_Survival.tick_params(axis='both',labelsize=6) 
ax1_2_Survival.plot(t_value, np.exp(-Lambda_Z0_X1), color='k', label='True') 
ax1_2_Survival.legend(loc='upper left', fontsize=6) 
plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.15)


# ------(prediction)---(Z=1,X=X1)------

fig2_Survival = plt.figure()
ax2_1_Survival = fig2_Survival.add_subplot(1, 2, 1)
ax2_1_Survival.set_title("Case 4, n=500, $\Lambda_0(t)=t^{3/4}/6$", fontsize=10) 
ax2_1_Survival.set_xlabel("Time",fontsize=8) 
ax2_1_Survival.set_ylabel("Survival function",fontsize=8) 
ax2_1_Survival.tick_params(axis='both',labelsize=6) 
ax2_1_Survival.plot(t_value, np.exp(-Lambda_Z1_X1), color='k', label='True') 
ax2_1_Survival.legend(loc='upper left', fontsize=6) 

ax2_2_Survival = fig2_Survival.add_subplot(1, 2, 2)
ax2_2_Survival.set_title("Case 4, n=1000, $\Lambda_0(t)=t^{3/4}/6$", fontsize=10) 
ax2_2_Survival.set_xlabel("Time",fontsize=8) 
ax2_2_Survival.set_ylabel("Survival function",fontsize=8) 
ax2_2_Survival.tick_params(axis='both',labelsize=6) 
ax2_2_Survival.plot(t_value, np.exp(-Lambda_Z1_X1), color='k', label='True') 
ax2_2_Survival.legend(loc='upper left', fontsize=6) 
plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.15)

# ------(prediction)---(Z=0,X=X2)------

fig3_Survival = plt.figure()
ax3_1_Survival = fig3_Survival.add_subplot(1, 2, 1)
ax3_1_Survival.set_title("Case 4, n=500, $\Lambda_0(t)=t^{3/4}/6$", fontsize=10) 
ax3_1_Survival.set_xlabel("Time",fontsize=8) 
ax3_1_Survival.set_ylabel("Survival function",fontsize=8) 
ax3_1_Survival.tick_params(axis='both',labelsize=6) 
ax3_1_Survival.plot(t_value, np.exp(-Lambda_Z0_X2), color='k', label='True') 
ax3_1_Survival.legend(loc='upper left', fontsize=6) 

ax3_2_Survival = fig3_Survival.add_subplot(1, 2, 2)
ax3_2_Survival.set_title("Case 4, n=1000, $\Lambda_0(t)=t^{3/4}/6$", fontsize=10) 
ax3_2_Survival.set_xlabel("Time",fontsize=8) 
ax3_2_Survival.set_ylabel("Survival function",fontsize=8) 
ax3_2_Survival.tick_params(axis='both',labelsize=6) 
ax3_2_Survival.plot(t_value, np.exp(-Lambda_Z0_X2), color='k', label='True') 
ax3_2_Survival.legend(loc='upper left', fontsize=6) 
plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.15)


# ------(prediction)---(Z=1,X=X2)------

fig4_Survival = plt.figure()
ax4_1_Survival = fig4_Survival.add_subplot(1, 2, 1)
ax4_1_Survival.set_title("Case 4, n=500, $\Lambda_0(t)=t^{3/4}/6$", fontsize=10) 
ax4_1_Survival.set_xlabel("Time",fontsize=8) 
ax4_1_Survival.set_ylabel("Survival function",fontsize=8) 
ax4_1_Survival.tick_params(axis='both',labelsize=6) 
ax4_1_Survival.plot(t_value, np.exp(-Lambda_Z1_X2), color='k', label='True') 
ax4_1_Survival.legend(loc='upper left', fontsize=6) 

ax4_2_Survival = fig4_Survival.add_subplot(1, 2, 2)
ax4_2_Survival.set_title("Case 4, n=1000, $\Lambda_0(t)=t^{3/4}/6$", fontsize=10) 
ax4_2_Survival.set_xlabel("Time",fontsize=8) 
ax4_2_Survival.set_ylabel("Survival function",fontsize=8) 
ax4_2_Survival.tick_params(axis='both',labelsize=6) 
ax4_2_Survival.plot(t_value, np.exp(-Lambda_Z1_X2), color='k', label='True') 
ax4_2_Survival.legend(loc='upper left', fontsize=6) 
plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0.3,hspace=0.15)

# ------(Estimation)---Bias,Sse,Ese,Cp------
beta_Bias_D = []; beta_Sse_D = []; beta_Ese_D = []; beta_Cp_D = []
gamma_Bias_D = []; gamma_Sse_D = []; gamma_Ese_D = []; gamma_Cp_D = []

beta_Bias_A = []; beta_Sse_A = []; beta_Ese_A = []; beta_Cp_A = []
gamma_Bias_A = []; gamma_Sse_A = []; gamma_Ese_A = []; gamma_Cp_A = []

beta_Bias_L = []; beta_Sse_L = []; beta_Ese_L = []; beta_Cp_L = []
gamma_Bias_L = []; gamma_Sse_L = []; gamma_Ese_L = []; gamma_Cp_L = []
# ------(Prediction)---------
G1_Re_D =[]; G1_sd_D =[]; G2_Re_D =[]; G2_sd_D =[]
G1_Re_A =[]; G1_sd_A =[]; G2_Re_A =[]; G2_sd_A =[]
G1_Re_L =[]; G1_sd_L =[]; G2_Re_L =[]; G2_sd_L =[]

for i in range(len(Set_n)):
    n = Set_n[i]
    n_node = Set_node[i]
    n_lr = Set_lr[i]
    #%% ------------ Store the results of B loops ---------
    g1_test_D = []; g2_test_D = []; C_D=[]; beta_D = []; gamma_D = []; Info_D = np.zeros((3,B)); g1_re_D = []; g2_re_D = []
    g1_test_A = []; g2_test_A = []; C_A=[]; beta_A = []; gamma_A = []; Info_A = np.zeros((3,B)); g1_re_A = []; g2_re_A = []
    g1_test_L = []; g2_test_L = []; C_L=[]; beta_L = []; gamma_L = []; Info_L = np.zeros((3,B)); g1_re_L = []; g2_re_L = []
    g1_subject_D = []; g2_subject_D = []
    g1_subject_A = []; g2_subject_A = []
    g1_subject_L = []; g2_subject_L = []
    for b in range(B):
        print('n=', n, 'b=', b)
        set_seed(12 + b)
        #%% ------------------------
        c_initial = np.array(0.1*np.ones(m+p), dtype="float32")
        if (i == 0):
            theta_initial = np.array([1.8, 0.95], dtype="float32")
        else:
            theta_initial = np.array([1.8, 0.9], dtype="float32")
        #%% ------------ Generate training data ------------
        train_data = generate_case_4(n, corr, theta0, tau)
        Z_train = train_data['Z']
        U_train = train_data['U']
        V_train = train_data['V']
        De1_train = train_data['De1']
        De2_train = train_data['De2']
        De3_train = train_data['De3']
        g1_train = train_data['g1_X']
        g2_train = train_data['g2_X']
        
        #%% =============DGAHM (Deep Generalized Additie Hazard Model)=============
        Est_D = Est_deep(train_data,X_test,X_subject,theta_initial,theta0,n_layer,n_node,n_lr,n_epoch,nodevec,m,c_initial)
        # -------Compute g_subject------
        g1_subject_D.append(Est_D['g_subject'][:,0]) 
        g2_subject_D.append(Est_D['g_subject'][:,1]) 
        # -------- About g_test---------
        g1_test_D.append(Est_D['g_test'][:,0]) 
        g2_test_D.append(Est_D['g_test'][:,1]) 
        # ------ Compute relative error and standard deviation ------
        g1_re_D.append(np.sqrt(np.mean((Est_D['g_test'][:,0]-np.mean(Est_D['g_test'][:,0])-g1_true)**2)/np.mean(g1_true**2)))
        g2_re_D.append(np.sqrt(np.mean((Est_D['g_test'][:,1]-np.mean(Est_D['g_test'][:,1])-g2_true)**2)/np.mean(g2_true**2)))
        # -------- About Lambda_U ---------
        C_D.append(Est_D['C']) 
        # ------- About the statistical inference of \hat\theta -----
        
        beta_D.append(Est_D['theta'][0])
        gamma_D.append(Est_D['theta'][1])
        
        Ezg1_D = np.exp(Z_train * Est_D['theta'][0] + Est_D['g_train'][:,0])
        Ezg2_D = np.exp(Z_train * Est_D['theta'][1] + Est_D['g_train'][:,1])
        
        Iu_D = I_U(m, np.clip(U_train * Ezg1_D, 0, 83), nodevec)
        Iv_D = I_U(m, np.clip(V_train * Ezg1_D, 0, 83), nodevec)
        Lamb_U_D = np.matmul(Iu_D, Est_D['C'])
        Lamb_V_D = np.matmul(Iv_D, Est_D['C'])
        f_U_D = Lamb_U_D * Ezg2_D
        f_V_D = Lamb_V_D * Ezg2_D
        
        Bu_D = B_S2(m, np.clip(U_train * Ezg1_D, 0, 83), nodevec)
        Bv_D = B_S2(m, np.clip(V_train * Ezg1_D, 0, 83), nodevec)
        dLamb_U_D = np.matmul(Bu_D, Est_D['C'])
        dLamb_V_D = np.matmul(Bv_D, Est_D['C']) 
        
        abc_beta_D = LFD_beta(train_data,Est_D['g_train'],Est_D['theta'],Est_D['C'],m,nodevec,n_layer,n_node=beta_node_D[i],n_lr=beta_lr_D[i],n_epoch=200)
        
        L_beta_D =  De1_train * U_train * Z_train * Ezg1_D * Ezg2_D * np.exp(-f_U_D) * dLamb_U_D / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * Z_train * Ezg1_D * Ezg2_D * (V_train * dLamb_V_D * np.exp(-f_V_D) - U_train * dLamb_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * V_train * Z_train * dLamb_V_D * Ezg1_D * Ezg2_D
        
        L_g1_beta_D = (De1_train * U_train * dLamb_U_D * Ezg1_D * Ezg2_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * Ezg1_D * Ezg2_D * (V_train * dLamb_V_D * np.exp(-f_V_D) - U_train * dLamb_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * V_train * dLamb_V_D * Ezg1_D * Ezg2_D) * abc_beta_D[:,0]
        
        L_g2_beta_D = (De1_train * f_U_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * (f_V_D * np.exp(-f_V_D)- f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * f_V_D) * abc_beta_D[:,1] 
        
        L_lambda_beta_D = De1_train * f_U_D * np.exp(-f_U_D) * abc_beta_D[:,2] / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * (abc_beta_D[:,3] * f_V_D * np.exp(-f_V_D)- abc_beta_D[:,2] * f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * f_V_D * abc_beta_D[:,3]
        
        
        abc_gamma_D = LFD_gamma(train_data,Est_D['g_train'],Est_D['theta'],Est_D['C'],m,nodevec,n_layer,n_node=gamma_node_D[i],n_lr=gamma_lr_D[i],n_epoch=200)
        
        L_gamma_D = De1_train * Z_train * f_U_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * Z_train * (f_V_D * np.exp(-f_V_D)- f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * Z_train * f_V_D
        
        L_g1_gamma_D = (De1_train * U_train * dLamb_U_D * Ezg1_D * Ezg2_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * Ezg1_D * Ezg2_D * (V_train * dLamb_V_D * np.exp(-f_V_D) - U_train * dLamb_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * V_train * dLamb_V_D * Ezg1_D * Ezg2_D) * abc_gamma_D[:,0]
        
        L_g2_gamma_D = (De1_train * f_U_D * np.exp(-f_U_D) / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * (f_V_D * np.exp(-f_V_D)- f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * f_V_D) * abc_gamma_D[:,1] 
        
        L_lambda_gamma_D = De1_train * f_U_D * np.exp(-f_U_D) * abc_gamma_D[:,2] / (1 - np.exp(-f_U_D) + 1e-4) + De2_train * (abc_gamma_D[:,3] * f_V_D * np.exp(-f_V_D)- abc_gamma_D[:,2] * f_U_D * np.exp(-f_U_D)) / (np.exp(-f_U_D) - np.exp(-f_V_D) + 1e-4) - De3_train * f_V_D * abc_gamma_D[:,3]
        
        I_1_D = L_beta_D - L_g1_beta_D - L_g2_beta_D - L_lambda_beta_D
        I_2_D = L_gamma_D - L_g1_gamma_D - L_g2_gamma_D - L_lambda_gamma_D
        
        Info_D[0, b] = np.mean(I_1_D ** 2)
        Info_D[1, b] = np.mean(I_2_D ** 2)
        Info_D[2, b] = np.mean(I_1_D * I_2_D)

        #%% =================== BS-GAHM(Generalized Additie Hazard Model by B-spline)================
        Est_A = Est_additive(train_data,X_test,X_subject,theta_initial,nodevec,m,c_initial,m0,nodevec0)
        # -------Compute g_subject------
        g1_subject_A.append(Est_A['g_subject'][:,0])
        g2_subject_A.append(Est_A['g_subject'][:,1])
        # -------- About g_test ---------
        g1_test_A.append(Est_A['g_test'][:,0])
        g2_test_A.append(Est_A['g_test'][:,1])
        # ------ Compute relative error and standard deviation ------
        g1_re_A.append(np.sqrt(np.mean((Est_A['g_test'][:,0]-np.mean(Est_A['g_test'][:,0])-g1_true)**2)/np.mean(g1_true**2)))
        g2_re_A.append(np.sqrt(np.mean((Est_A['g_test'][:,1]-np.mean(Est_A['g_test'][:,1])-g2_true)**2)/np.mean(g2_true**2)))
        # -------- About Lambda_U ---------
        C_A.append(Est_A['C']) 
        # -------------------------------------------
        beta_A.append(Est_A['theta'][0])
        gamma_A.append(Est_A['theta'][1])
        
        Ezg1_A = np.exp(Z_train * Est_A['theta'][0] + Est_A['g_train'][:,0])
        Ezg2_A = np.exp(Z_train * Est_A['theta'][1] + Est_A['g_train'][:,1])
        # compute \Lambda()
        Iu_A = I_U(m, np.clip(U_train * Ezg1_A, 0, 83), nodevec)
        Iv_A = I_U(m, np.clip(V_train * Ezg1_A, 0, 83), nodevec)
        Lamb_U_A = np.matmul(Iu_A, Est_A['C'])
        Lamb_V_A = np.matmul(Iv_A, Est_A['C'])
        f_U_A = Lamb_U_A * Ezg2_A
        f_V_A = Lamb_V_A * Ezg2_A
        # compute \Lambda'()
        Bu_A = B_S2(m, np.clip(U_train * Ezg1_A, 0, 83), nodevec)
        Bv_A = B_S2(m, np.clip(V_train * Ezg1_A, 0, 83), nodevec)
        dLamb_U_A = np.matmul(Bu_A, Est_A['C'])
        dLamb_V_A = np.matmul(Bv_A, Est_A['C']) 
        
        abc_beta_A = LFD_beta(train_data,Est_A['g_train'],Est_A['theta'],Est_A['C'],m,nodevec,n_layer,n_node=beta_node_A[i],n_lr=beta_lr_A[i],n_epoch=200)
        
        L_beta_A =  De1_train * U_train * Z_train * Ezg1_A * Ezg2_A * np.exp(-f_U_A) * dLamb_U_A / (1 - np.exp(-f_U_A) + 1e-4) + De2_train * Z_train * Ezg1_A * Ezg2_A * (V_train * dLamb_V_A * np.exp(-f_V_A) - U_train * dLamb_U_A * np.exp(-f_U_A)) / (np.exp(-f_U_A) - np.exp(-f_V_A) + 1e-4) - De3_train * V_train * Z_train * dLamb_V_A * Ezg1_A * Ezg2_A
        
        L_g1_beta_A = (De1_train * U_train * dLamb_U_A * Ezg1_A * Ezg2_A * np.exp(-f_U_A) / (1 - np.exp(-f_U_A) + 1e-4) + De2_train * Ezg1_A * Ezg2_A * (V_train * dLamb_V_A * np.exp(-f_V_A) - U_train * dLamb_U_A * np.exp(-f_U_A)) / (np.exp(-f_U_A) - np.exp(-f_V_A) + 1e-4) - De3_train * V_train * dLamb_V_A * Ezg1_A * Ezg2_A) * abc_beta_A[:,0]
        
        L_g2_beta_A = (De1_train * f_U_A * np.exp(-f_U_A) / (1 - np.exp(-f_U_A) + 1e-4) + De2_train * (f_V_A * np.exp(-f_V_A)- f_U_A * np.exp(-f_U_A)) / (np.exp(-f_U_A) - np.exp(-f_V_A) + 1e-4) - De3_train * f_V_A) * abc_beta_A[:,1] 
        
        L_lambda_beta_A = De1_train * f_U_A * np.exp(-f_U_A) * abc_beta_A[:,2] / (1 - np.exp(-f_U_A) + 1e-4) + De2_train * (abc_beta_A[:,3] * f_V_A * np.exp(-f_V_A)- abc_beta_A[:,2] * f_U_A * np.exp(-f_U_A)) / (np.exp(-f_U_A) - np.exp(-f_V_A) + 1e-4) - De3_train * f_V_A * abc_beta_A[:,3]
        
        
        abc_gamma_A = LFD_gamma(train_data,Est_A['g_train'],Est_A['theta'],Est_A['C'],m,nodevec,n_layer,n_node=gamma_node_A[i],n_lr=gamma_lr_A[i],n_epoch=200)
        
        L_gamma_A = De1_train * Z_train * f_U_A * np.exp(-f_U_A) / (1 - np.exp(-f_U_A) + 1e-4) + De2_train * Z_train * (f_V_A * np.exp(-f_V_A)- f_U_A * np.exp(-f_U_A)) / (np.exp(-f_U_A) - np.exp(-f_V_A) + 1e-4) - De3_train * Z_train * f_V_A
        
        L_g1_gamma_A = (De1_train * U_train * dLamb_U_A * Ezg1_A * Ezg2_A * np.exp(-f_U_A) / (1 - np.exp(-f_U_A) + 1e-4) + De2_train * Ezg1_A * Ezg2_A * (V_train * dLamb_V_A * np.exp(-f_V_A) - U_train * dLamb_U_A * np.exp(-f_U_A)) / (np.exp(-f_U_A) - np.exp(-f_V_A) + 1e-4) - De3_train * V_train * dLamb_V_A * Ezg1_A * Ezg2_A) * abc_gamma_A[:,0]
        
        L_g2_gamma_A = (De1_train * f_U_A * np.exp(-f_U_A) / (1 - np.exp(-f_U_A) + 1e-4) + De2_train * (f_V_A * np.exp(-f_V_A)- f_U_A * np.exp(-f_U_A)) / (np.exp(-f_U_A) - np.exp(-f_V_A) + 1e-4) - De3_train * f_V_A) * abc_gamma_A[:,1] 
        
        L_lambda_gamma_A = De1_train * f_U_A * np.exp(-f_U_A) * abc_gamma_A[:,2] / (1 - np.exp(-f_U_A) + 1e-4) + De2_train * (abc_gamma_A[:,3] * f_V_A * np.exp(-f_V_A)- abc_gamma_A[:,2] * f_U_A * np.exp(-f_U_A)) / (np.exp(-f_U_A) - np.exp(-f_V_A) + 1e-4) - De3_train * f_V_A * abc_gamma_A[:,3]
        
        I_1_A = L_beta_A - L_g1_beta_A - L_g2_beta_A - L_lambda_beta_A
        I_2_A = L_gamma_A - L_g1_gamma_A - L_g2_gamma_A - L_lambda_gamma_A
        
        Info_A[0, b] = np.mean(I_1_A ** 2)
        Info_A[1, b] = np.mean(I_2_A ** 2)
        Info_A[2, b] = np.mean(I_1_A * I_2_A) 
        #%% ===================GHM(General Hazards Model)================
        Est_L = Est_linear(train_data,X_test,X_subject,theta_initial,nodevec,m,c_initial)
        # -------Compute g_subject------
        g1_subject_L.append(Est_L['g_subject'][:,0]) 
        g2_subject_L.append(Est_L['g_subject'][:,1]) 
        # -------- About g_test ---------
        g1_test_L.append(Est_L['g_test'][:,0]) 
        g2_test_L.append(Est_L['g_test'][:,1]) 
        # ------ Compute relative error and standard deviation ------
        g1_re_L.append(np.sqrt(np.mean((Est_L['g_test'][:,0]-np.mean(Est_L['g_test'][:,0])-g1_true)**2)/np.mean(g1_true**2)))
        g2_re_L.append(np.sqrt(np.mean((Est_L['g_test'][:,1]-np.mean(Est_L['g_test'][:,1])-g2_true)**2)/np.mean(g2_true**2)))
        # -------- About Lambda_U ---------
        C_L.append(Est_L['C'])
        # ------- About the statistical inference of \hat\theta -----
        
        beta_L.append(Est_L['theta'][0])
        gamma_L.append(Est_L['theta'][1])
        
        Ezg1_L = np.exp(Z_train * Est_L['theta'][0] + Est_L['g_train'][:,0])
        Ezg2_L = np.exp(Z_train * Est_L['theta'][1] + Est_L['g_train'][:,1])
        
        Iu_L = I_U(m, np.clip(U_train * Ezg1_L, 0, 83), nodevec)
        Iv_L = I_U(m, np.clip(V_train * Ezg1_L, 0, 83), nodevec)
        Lamb_U_L = np.matmul(Iu_L, Est_L['C'])
        Lamb_V_L = np.matmul(Iv_L, Est_L['C'])
        f_U_L = Lamb_U_L * Ezg2_L
        f_V_L = Lamb_V_L * Ezg2_L
        
        Bu_L = B_S2(m, np.clip(U_train * Ezg1_L, 0, 83), nodevec)
        Bv_L = B_S2(m, np.clip(V_train * Ezg1_L, 0, 83), nodevec)
        dLamb_U_L = np.matmul(Bu_L, Est_L['C'])
        dLamb_V_L = np.matmul(Bv_L, Est_L['C']) 
        
        abc_beta_L = LFD_beta(train_data,Est_L['g_train'],Est_L['theta'],Est_L['C'],m,nodevec,n_layer,n_node=beta_node_L[i],n_lr=beta_lr_L[i],n_epoch=200)
        
        L_beta_L =  De1_train * U_train * Z_train * Ezg1_L * Ezg2_L * np.exp(-f_U_L) * dLamb_U_L / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * Z_train * Ezg1_L * Ezg2_L * (V_train * dLamb_V_L * np.exp(-f_V_L) - U_train * dLamb_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * V_train * Z_train * dLamb_V_L * Ezg1_L * Ezg2_L
        
        L_g1_beta_L = (De1_train * U_train * dLamb_U_L * Ezg1_L * Ezg2_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * Ezg1_L * Ezg2_L * (V_train * dLamb_V_L * np.exp(-f_V_L) - U_train * dLamb_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * V_train * dLamb_V_L * Ezg1_L * Ezg2_L) * abc_beta_L[:,0]
        
        L_g2_beta_L = (De1_train * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * f_V_L) * abc_beta_L[:,1] 
        
        L_lambda_beta_L = De1_train * f_U_L * np.exp(-f_U_L) * abc_beta_L[:,2] / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * (abc_beta_L[:,3] * f_V_L * np.exp(-f_V_L)- abc_beta_L[:,2] * f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * f_V_L * abc_beta_L[:,3]
        
        
        abc_gamma_L = LFD_gamma(train_data,Est_L['g_train'],Est_L['theta'],Est_L['C'],m,nodevec,n_layer,n_node=gamma_node_L[i],n_lr=gamma_lr_L[i],n_epoch=200)
        
        L_gamma_L = De1_train * Z_train * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * Z_train * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * Z_train * f_V_L
        
        L_g1_gamma_L = (De1_train * U_train * dLamb_U_L * Ezg1_L * Ezg2_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * Ezg1_L * Ezg2_L * (V_train * dLamb_V_L * np.exp(-f_V_L) - U_train * dLamb_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * V_train * dLamb_V_L * Ezg1_L * Ezg2_L) * abc_gamma_L[:,0]
        
        L_g2_gamma_L = (De1_train * f_U_L * np.exp(-f_U_L) / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * (f_V_L * np.exp(-f_V_L)- f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * f_V_L) * abc_gamma_L[:,1] 
        
        L_lambda_gamma_L = De1_train * f_U_L * np.exp(-f_U_L) * abc_gamma_L[:,2] / (1 - np.exp(-f_U_L) + 1e-4) + De2_train * (abc_gamma_L[:,3] * f_V_L * np.exp(-f_V_L)- abc_gamma_L[:,2] * f_U_L * np.exp(-f_U_L)) / (np.exp(-f_U_L) - np.exp(-f_V_L) + 1e-4) - De3_train * f_V_L * abc_gamma_L[:,3]
        
        I_1_L = L_beta_L - L_g1_beta_L - L_g2_beta_L - L_lambda_beta_L
        I_2_L = L_gamma_L - L_g1_gamma_L - L_g2_gamma_L - L_lambda_gamma_L
        
        Info_L[0, b] = np.mean(I_1_L ** 2)
        Info_L[1, b] = np.mean(I_2_L ** 2)
        Info_L[2, b] = np.mean(I_1_L * I_2_L) 
    #%% =============DGAHM (Deep Generalized Additie Hazard Model)=============
    # =====================Figures=====================
    Error_g1_D = np.mean(np.array(g1_test_D), axis=0) - g1_true
    Error_g2_D = np.mean(np.array(g2_test_D), axis=0) - g2_true

    g1_subject_D = np.array(g1_subject_D)
    g2_subject_D = np.array(g2_subject_D)
    if (i == 0):
        # prediction for S(t|X=X1,Z=0)
        ax1_1_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_D), axis=0), t_value*np.exp(np.mean(np.array(g1_subject_D[:,0]))),nodevec) * np.exp(np.mean(np.array(g2_subject_D[:,0])))), label='DGAHM', linestyle=':')
        ax1_1_Survival.legend(loc='upper left', fontsize=6)
        # prediction for S(t|X=X1,Z=1)
        ax2_1_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_D), axis=0), t_value*np.exp(np.mean(np.array(beta_D)) + np.mean(np.array(g1_subject_D[:,0]))),nodevec) * np.exp(np.mean(np.array(gamma_D)) + np.mean(np.array(g2_subject_D[:,0])))), label='DGAHM', linestyle=':')
        ax2_1_Survival.legend(loc='upper left', fontsize=6)
        # prediction for S(t|X=X2,Z=0)
        ax3_1_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_D), axis=0), t_value*np.exp(np.mean(np.array(g1_subject_D[:,1]))),nodevec) * np.exp(np.mean(np.array(g2_subject_D[:,1])))), label='DGAHM', linestyle=':')
        ax3_1_Survival.legend(loc='upper left', fontsize=6)
        # prediction for S(t|X=X2,Z=1)
        ax4_1_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_D), axis=0), t_value*np.exp(np.mean(np.array(beta_D)) + np.mean(np.array(g1_subject_D[:,1]))),nodevec) * np.exp(np.mean(np.array(gamma_D)) + np.mean(np.array(g2_subject_D[:,1])))), label='DGAHM', linestyle=':')
        ax4_1_Survival.legend(loc='upper left', fontsize=6)
    else:
        # prediction for S(t|X=X1,Z=0)
        ax1_2_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_D), axis=0), t_value*np.exp(np.mean(np.array(g1_subject_D[:,0]))),nodevec) * np.exp(np.mean(np.array(g2_subject_D[:,0])))), label='DGAHM', linestyle=':')
        ax1_2_Survival.legend(loc='upper left', fontsize=6)
        # prediction for S(t|X=X1,Z=1)
        ax2_2_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_D), axis=0), t_value*np.exp(np.mean(np.array(beta_D)) + np.mean(np.array(g1_subject_D[:,0]))),nodevec) * np.exp(np.mean(np.array(gamma_D)) + np.mean(np.array(g2_subject_D[:,0])))), label='DGAHM', linestyle=':')
        ax2_2_Survival.legend(loc='upper left', fontsize=6)
        # prediction for S(t|X=X2,Z=0)
        ax3_2_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_D), axis=0), t_value*np.exp(np.mean(np.array(g1_subject_D[:,1]))),nodevec) * np.exp(np.mean(np.array(g2_subject_D[:,1])))), label='DGAHM', linestyle=':')
        ax3_2_Survival.legend(loc='upper left', fontsize=6)
        # prediction for S(t|X=X2,Z=1)
        ax4_2_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_D), axis=0), t_value*np.exp(np.mean(np.array(beta_D)) + np.mean(np.array(g1_subject_D[:,1]))),nodevec) * np.exp(np.mean(np.array(gamma_D)) + np.mean(np.array(g2_subject_D[:,1])))), label='DGAHM', linestyle=':')
        ax4_2_Survival.legend(loc='upper left', fontsize=6)


    # ==================Bias, Sse, Ese, Cp of (beta,gamma)==================
    # save Bias, Sse
    beta_Bias_D.append(np.mean(np.array(beta_D))-theta0[0])
    beta_Sse_D.append(np.sqrt(np.mean((np.array(beta_D)-np.mean(np.array(beta_D)))**2)))
    gamma_Bias_D.append(np.mean(np.array(gamma_D))-theta0[1])
    gamma_Sse_D.append(np.sqrt(np.mean((np.array(gamma_D)-np.mean(np.array(gamma_D)))**2)))
    # compute Information Matrix
    IM_D = np.zeros((2,2))
    IM_D[0,0] = np.mean(Info_D[0])
    IM_D[1,1] = np.mean(Info_D[1])
    IM_D[0,1] = np.mean(Info_D[2])
    IM_D[1,0] = IM_D[0,1]
    # compute Covariance Matrix
    Sigma_D = np.linalg.inv(IM_D)/n
    sd_beta_D = np.sqrt(Sigma_D[0,0]) 
    sd_gamma_D = np.sqrt(Sigma_D[1,1]) 
    # save Ese
    beta_Ese_D.append(sd_beta_D)
    gamma_Ese_D.append(sd_gamma_D)
    # save Cp
    beta_Cp_D.append(np.mean((np.array(beta_D) - 1.96 * sd_beta_D <= theta0[0]) * (theta0[0] <= np.array(beta_D) + 1.96 * sd_beta_D)))
    gamma_Cp_D.append(np.mean((np.array(gamma_D) - 1.96 * sd_gamma_D <= theta0[1]) * (theta0[1] <= np.array(gamma_D) + 1.96 * sd_gamma_D)))
    # ===================Re and Sd of g1, g2=================
    G1_Re_D.append(np.mean(g1_re_D))
    G1_sd_D.append(np.sqrt(np.mean((g1_re_D-np.mean(g1_re_D))**2)))
    G2_Re_D.append(np.mean(g2_re_D))
    G2_sd_D.append(np.sqrt(np.mean((g2_re_D-np.mean(g2_re_D))**2)))
    
    #%% =================== BS-GAHM(Generalized Additie Hazard Model by B-spline)================
    # =====================Figures====================
    Error_g1_A = np.mean(np.array(g1_test_A), axis=0) - g1_true
    Error_g2_A = np.mean(np.array(g2_test_A), axis=0) - g2_true

    g1_subject_A = np.array(g1_subject_A)
    g2_subject_A = np.array(g2_subject_A)
    if (i == 0):
        # prediction for S(t|X=X1,Z=0)
        ax1_1_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_A), axis=0), t_value*np.exp(np.mean(np.array(g1_subject_A[:,0]))),nodevec) * np.exp(np.mean(np.array(g2_subject_A[:,0])))), label='BS-GAHM', linestyle='-.')
        ax1_1_Survival.legend(loc='upper left', fontsize=6)
        # prediction for S(t|X=X1,Z=1)
        ax2_1_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_A), axis=0), t_value*np.exp(np.mean(np.array(beta_A)) + np.mean(np.array(g1_subject_A[:,0]))),nodevec) * np.exp(np.mean(np.array(gamma_A)) + np.mean(np.array(g2_subject_A[:,0])))), label='BS-GAHM', linestyle='-.')
        ax2_1_Survival.legend(loc='upper left', fontsize=6)
        # prediction for S(t|X=X2,Z=0)
        ax3_1_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_A), axis=0), t_value*np.exp(np.mean(np.array(g1_subject_A[:,1]))),nodevec) * np.exp(np.mean(np.array(g2_subject_A[:,1])))), label='BS-GAHM', linestyle='-.')
        ax3_1_Survival.legend(loc='upper left', fontsize=6)
        # prediction for S(t|X=X2,Z=1)
        ax4_1_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_A), axis=0), t_value*np.exp(np.mean(np.array(beta_A)) + np.mean(np.array(g1_subject_A[:,1]))),nodevec) * np.exp(np.mean(np.array(gamma_A)) + np.mean(np.array(g2_subject_A[:,1])))), label='BS-GAHM', linestyle='-.')
        ax4_1_Survival.legend(loc='upper left', fontsize=6)
    else:
        # prediction for S(t|X=X1,Z=0)
        ax1_2_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_A), axis=0), t_value*np.exp(np.mean(np.array(g1_subject_A[:,0]))),nodevec) * np.exp(np.mean(np.array(g2_subject_A[:,0])))), label='BS-GAHM', linestyle='-.')
        ax1_2_Survival.legend(loc='upper left', fontsize=6)
        # prediction for S(t|X=X1,Z=1)
        ax2_2_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_A), axis=0), t_value*np.exp(np.mean(np.array(beta_A)) + np.mean(np.array(g1_subject_A[:,0]))),nodevec) * np.exp(np.mean(np.array(gamma_A)) + np.mean(np.array(g2_subject_A[:,0])))), label='BS-GAHM', linestyle='-.')
        ax2_2_Survival.legend(loc='upper left', fontsize=6)
        # prediction for S(t|X=X2,Z=0)
        ax3_2_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_A), axis=0), t_value*np.exp(np.mean(np.array(g1_subject_A[:,1]))),nodevec) * np.exp(np.mean(np.array(g2_subject_A[:,1])))), label='BS-GAHM', linestyle='-.')
        ax3_2_Survival.legend(loc='upper left', fontsize=6)
        # prediction for S(t|X=X2,Z=1)
        ax4_2_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_A), axis=0), t_value*np.exp(np.mean(np.array(beta_A)) + np.mean(np.array(g1_subject_A[:,1]))),nodevec) * np.exp(np.mean(np.array(gamma_A)) + np.mean(np.array(g2_subject_A[:,1])))), label='BS-GAHM', linestyle='-.')
        ax4_2_Survival.legend(loc='upper left', fontsize=6)
    # ===================Bias, Sse, Ese, Cp of (beta,gamma)==================
    # save Bias, Sse
    beta_Bias_A.append(np.mean(np.array(beta_A))-theta0[0])
    beta_Sse_A.append(np.sqrt(np.mean((np.array(beta_A)-np.mean(np.array(beta_A)))**2)))
    gamma_Bias_A.append(np.mean(np.array(gamma_A))-theta0[1])
    gamma_Sse_A.append(np.sqrt(np.mean((np.array(gamma_A)-np.mean(np.array(gamma_A)))**2)))
    # compute Information Matrix
    IM_A = np.zeros((2,2))
    IM_A[0,0] = np.mean(Info_A[0])
    IM_A[1,1] = np.mean(Info_A[1])
    IM_A[0,1] = np.mean(Info_A[2])
    IM_A[1,0] = IM_A[0,1]
    # compute Covariance Matrix
    Sigma_A = np.linalg.inv(IM_A)/n
    sd_beta_A = np.sqrt(Sigma_A[0,0]) 
    sd_gamma_A = np.sqrt(Sigma_A[1,1]) 
    # save Ese
    beta_Ese_A.append(sd_beta_A)
    gamma_Ese_A.append(sd_gamma_A)
    # save Cp
    beta_Cp_A.append(np.mean((np.array(beta_A) - 1.96 * sd_beta_A <= theta0[0]) * (theta0[0] <= np.array(beta_A) + 1.96 * sd_beta_A)))
    gamma_Cp_A.append(np.mean((np.array(gamma_A) - 1.96 * sd_gamma_A <= theta0[1]) * (theta0[1] <= np.array(gamma_A) + 1.96 * sd_gamma_A)))
    #==================Re and Sd of g1, g2==================
    G1_Re_A.append(np.mean(g1_re_A))
    G1_sd_A.append(np.sqrt(np.mean((g1_re_A-np.mean(g1_re_A))**2)))
    G2_Re_A.append(np.mean(g2_re_A))
    G2_sd_A.append(np.sqrt(np.mean((g2_re_A-np.mean(g2_re_A))**2)))

    #%% ===================GHM(General Hazards Model)================
    #=====================Figures====================
    Error_g1_L = np.mean(np.array(g1_test_L), axis=0) - g1_true
    Error_g2_L = np.mean(np.array(g2_test_L), axis=0) - g2_true

    g1_subject_L = np.array(g1_subject_L)
    g2_subject_L = np.array(g2_subject_L)
    if (i == 0):
        # prediction for S(t|X=X1,Z=0)
        ax1_1_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_L), axis=0), t_value*np.exp(np.mean(np.array(g1_subject_L[:,0]))),nodevec) * np.exp(np.mean(np.array(g2_subject_L[:,0])))), label='GHM', linestyle='--')
        ax1_1_Survival.legend(loc='upper left', fontsize=6)
        # prediction for S(t|X=X1,Z=1)
        ax2_1_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_L), axis=0), t_value*np.exp(np.mean(np.array(beta_L)) + np.mean(np.array(g1_subject_L[:,0]))),nodevec) * np.exp(np.mean(np.array(gamma_L)) + np.mean(np.array(g2_subject_L[:,0])))), label='GHM', linestyle='--')
        ax2_1_Survival.legend(loc='upper left', fontsize=6)
        # prediction for S(t|X=X2,Z=0)
        ax3_1_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_L), axis=0), t_value*np.exp(np.mean(np.array(g1_subject_L[:,1]))),nodevec) * np.exp(np.mean(np.array(g2_subject_L[:,1])))), label='GHM', linestyle='--')
        ax3_1_Survival.legend(loc='upper left', fontsize=6)
        # prediction for S(t|X=X2,Z=1)
        ax4_1_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_L), axis=0), t_value*np.exp(np.mean(np.array(beta_L)) + np.mean(np.array(g1_subject_L[:,1]))),nodevec) * np.exp(np.mean(np.array(gamma_L)) + np.mean(np.array(g2_subject_L[:,1])))), label='GHM', linestyle='--')
        ax4_1_Survival.legend(loc='upper left', fontsize=6)
    else:
        # prediction for S(t|X=X1,Z=0)
        ax1_2_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_L), axis=0), t_value*np.exp(np.mean(np.array(g1_subject_L[:,0]))),nodevec) * np.exp(np.mean(np.array(g2_subject_L[:,0])))), label='GHM', linestyle='--')
        ax1_2_Survival.legend(loc='upper left', fontsize=6)
        # prediction for S(t|X=X1,Z=1)
        ax2_2_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_L), axis=0), t_value*np.exp(np.mean(np.array(beta_L)) + np.mean(np.array(g1_subject_L[:,0]))),nodevec) * np.exp(np.mean(np.array(gamma_L)) + np.mean(np.array(g2_subject_L[:,0])))), label='GHM', linestyle='--')
        ax2_2_Survival.legend(loc='upper left', fontsize=6)
        # prediction for S(t|X=X2,Z=0)
        ax3_2_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_L), axis=0), t_value*np.exp(np.mean(np.array(g1_subject_L[:,1]))),nodevec) * np.exp(np.mean(np.array(g2_subject_L[:,1])))), label='GHM', linestyle='--')
        ax3_2_Survival.legend(loc='upper left', fontsize=6)
        # prediction for S(t|X=X2,Z=1)
        ax4_2_Survival.plot(t_value, np.exp(-I_S(m,np.mean(np.array(C_L), axis=0), t_value*np.exp(np.mean(np.array(beta_L)) + np.mean(np.array(g1_subject_L[:,1]))),nodevec) * np.exp(np.mean(np.array(gamma_L)) + np.mean(np.array(g2_subject_L[:,1])))), label='GHM', linestyle='--')
        ax4_2_Survival.legend(loc='upper left', fontsize=6)
    # ===================Bias, Sse, Ese, Cp of (beta,gamma)==================
    # save Bias, Sse
    beta_Bias_L.append(np.mean(np.array(beta_L))-theta0[0])
    beta_Sse_L.append(np.sqrt(np.mean((np.array(beta_L)-np.mean(np.array(beta_L)))**2)))
    gamma_Bias_L.append(np.mean(np.array(gamma_L))-theta0[1])
    gamma_Sse_L.append(np.sqrt(np.mean((np.array(gamma_L)-np.mean(np.array(gamma_L)))**2)))
    # compute Information Matrix
    IM_L = np.zeros((2,2))
    IM_L[0,0] = np.mean(Info_L[0])
    IM_L[1,1] = np.mean(Info_L[1])
    IM_L[0,1] = np.mean(Info_L[2])
    IM_L[1,0] = IM_L[0,1]
    # compute Covariance Matrix
    Sigma_L = np.linalg.inv(IM_L)/n
    sd_beta_L = np.sqrt(Sigma_L[0,0]) 
    sd_gamma_L = np.sqrt(Sigma_L[1,1])
    # save Ese
    beta_Ese_L.append(sd_beta_L)
    gamma_Ese_L.append(sd_gamma_L)
    # save Cp
    beta_Cp_L.append(np.mean((np.array(beta_L) - 1.96 * sd_beta_L <= theta0[0]) * (theta0[0] <= np.array(beta_L) + 1.96 * sd_beta_L)))
    gamma_Cp_L.append(np.mean((np.array(gamma_L) - 1.96 * sd_gamma_L <= theta0[1]) * (theta0[1] <= np.array(gamma_L) + 1.96 * sd_gamma_L)))
    #==================Re and Sd of g1, g2==================
    G1_Re_L.append(np.mean(g1_re_L))
    G1_sd_L.append(np.sqrt(np.mean((g1_re_L-np.mean(g1_re_L))**2)))
    G2_Re_L.append(np.mean(g2_re_L))
    G2_sd_L.append(np.sqrt(np.mean((g2_re_L-np.mean(g2_re_L))**2)))
#%% -----------Save all results------------
# ================Figures======================
fig1_Survival.savefig('fig_Survival_Z0_X1_case4(non-linear).jpeg', dpi=400, bbox_inches='tight')
fig2_Survival.savefig('fig_Survival_Z1_X1_case4(non-linear).jpeg', dpi=400, bbox_inches='tight')
fig3_Survival.savefig('fig_Survival_Z0_X2_case4(non-linear).jpeg', dpi=400, bbox_inches='tight')
fig4_Survival.savefig('fig_Survival_Z1_X2_case4(non-linear).jpeg', dpi=400, bbox_inches='tight')
# =================Tables=======================
# results for beta
dic_error_beta = {"n": Set_n, "beta_Bias_D": np.array(beta_Bias_D), "beta_Sse_D": np.array(beta_Sse_D), "beta_Ese_D": np.array(beta_Ese_D), "beta_Cp_D": np.array(beta_Cp_D), "beta_Bias_A": np.array(beta_Bias_A), "beta_Sse_A": np.array(beta_Sse_A), "beta_Ese_A": np.array(beta_Ese_A), "beta_Cp_A": np.array(beta_Cp_A),"beta_Bias_L": np.array(beta_Bias_L), "beta_Sse_L": np.array(beta_Sse_L), "beta_Ese_L": np.array(beta_Ese_L), "beta_Cp_L": np.array(beta_Cp_L)}
result_error_beta = pd.DataFrame(dic_error_beta)
result_error_beta.to_csv('beta_error_case4(non-linear).csv')
# results for beta
dic_error_gamma = {"n": Set_n, "gamma_Bias_D": np.array(gamma_Bias_D), "gamma_Sse_D": np.array(gamma_Sse_D), "gamma_Ese_D": np.array(gamma_Ese_D), "gamma_Cp_D": np.array(gamma_Cp_D), "gamma_Bias_A": np.array(gamma_Bias_A), "gamma_Sse_A": np.array(gamma_Sse_A), "gamma_Ese_A": np.array(gamma_Ese_A), "gamma_Cp_A": np.array(gamma_Cp_A), "gamma_Bias_L": np.array(gamma_Bias_L), "gamma_Sse_L": np.array(gamma_Sse_L), "gamma_Ese_L": np.array(gamma_Ese_L), "gamma_Cp_L": np.array(gamma_Cp_L)}
result_error_gamma = pd.DataFrame(dic_error_gamma)
result_error_gamma.to_csv('gamma_error_case4(non-linear).csv')
# results for g1
dic_re_g1 = {"n": Set_n, "G1_Re_D": np.array(G1_Re_D), "G1_sd_D": np.array(G1_sd_D), "G1_Re_A": np.array(G1_Re_A), "G1_sd_A": np.array(G1_sd_A),"G1_Re_L": np.array(G1_Re_L), "G1_sd_L": np.array(G1_sd_L)}
result_re_g1 = pd.DataFrame(dic_re_g1)
result_re_g1.to_csv('Re_g1_case4(non-linear).csv')
# results for g2
dic_re_g2 = {"n": Set_n, "G2_Re_D": np.array(G2_Re_D), "G2_sd_D": np.array(G2_sd_D), "G2_Re_A": np.array(G2_Re_A), "G2_sd_A": np.array(G2_sd_A),"G2_Re_L": np.array(G2_Re_L), "G2_sd_L": np.array(G2_sd_L)}
result_re_g2 = pd.DataFrame(dic_re_g2)
result_re_g2.to_csv('Re_g2_case4(non-linear).csv')
