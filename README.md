# DGAHM_Interval_Censored
# DNN_case1_interval

This is the implementation of **Deep Partially Linear Cox Model for Current Status Data**. In the simulations, we consider four cases about $g_0(X)$. The four folders(**Model_Linear, Model_Additive, Model_Deep1, Model_Deep2**) present the code for the corresponding cases. The folder **Application** shows the results for the real data, which can be publicly available at https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity.

## Reference to download and install (Python 3.10.8)

+ pip install packages

> pip install numpy pandas torch matplotlib scipy


## Usage
> First, given the initial values of $\beta$ and spline parameters $c$, we can obtain the parameter estimates in the neural network by minimizing the negative log-likelihood function using ***PyTorch***, and bring them into the negative log-likelihood function, and then minimizing the negative log-likelihood function w.r.t $(\beta, c)$, thereby obtaining parameter estimates of $\beta$ and $c$, and loop iteration until the parameters converge.

1. How to optimize parameters with constraints?
> We use ***minimize***, which is a function of the ***optimize*** module in ***scipy***

```
import scipy.optimize as opt
res=opt.minimize()

res=opt.minimize(fun, x0, args=(), method=None, jac=None, hess=None,
              hessp=None, bounds=None, constraints=(), tol=None,
              callback=None, options=None)
#fun: This parameter is the costFunction loss function you want to minimize. Pass the name of costFunction to fun.
#x0: guessed initial value
#args=(): Additional parameters for optimization, starting from the second one by default
#method: This parameter represents the method used. The default is one of BFGS, L-BFGS-B, and SLSQP.
#options: Used to control the maximum number of iterations, set in the form of a dictionary, for example: options={‘maxiter’:400}
#constraints: Constraints, restricting the part of fun that is a parameter. Multiple constraints are as follows:
           '''cons = ({'type': 'ineq', 'fun': lambda x: x[0] - x1min},\
              {'type': 'ineq', 'fun': lambda x: -x[0] + x1max},\
              {'type': 'ineq', 'fun': lambda x: x[1] - x2min},\
              {'type': 'ineq', 'fun': lambda x: -x[1] + x2max})'''
#tol: Objective function error range, control the end of iteration
#callback: Preserve optimization process
```

***Example***

```
import numpy as np
import scipy.optimize as opt
def f(x):
    return (x + 2*np.cos(x) + 2)**2
bnds = [(0,4)]
res=opt.minimize(f, x0=1, method='SLSQP', bounds=bnds)
res
```

2. How to train a neural network?
> We use **PyTorch**.

```
    class DNNModel(torch.nn.Module):
        def __init__(self):
            super(DNNModel, self).__init__()
            layers = []
            layers.append(nn.Linear(n_input, n_node))
            layers.append(nn.ReLU())
            for i in range(n_layer):
                layers.append(nn.Linear(n_node, n_node))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(n_node, n_output))
            self.model = nn.Sequential(*layers)
        def forward(self, x):
            y_pred = self.model(x)
            return y_pred
    model = DNNModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=n_lr)
    for epoch in range(n_epoch):
        pred_g_X = model(X_train)
        loss = my_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    g_train = model(X_train)
```

