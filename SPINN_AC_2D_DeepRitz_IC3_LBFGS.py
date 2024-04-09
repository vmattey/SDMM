#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/vmattey/SeparablePINN_AC_Codes/blob/main/SPINN_AC_2D_DeepRitz_IC2.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Code for Separable PINN in PyTorch
# Solving 2D Allen Cahn Equation using separable Deep Ritz
# 
# # Details of the equation
# $u_t = ∇^2u - (1/ϵ^2)*(u^2 - 1)*u$
# 
# *   $x,y = [0,1] × [0,1] $, $t = [0, 0.02]$
# *   $ϵ = 0.01$
# *   BC: grad(u).n = 0 ---> No flux on the boundary
# *   Initial Condition: $u_0 = \frac{\tanh(R_0 + 0.1 \cos(7 \theta) - \sqrt{(x-0.5)^2 + (y-0.5)^2})}{\sqrt{2}\epsilon}$
# 
# 

# In[1]:


import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch.func as ft
import numpy as np
import time
import torch.jit as jit
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import torch.optim.lr_scheduler as lr_scheduler
# Seed for randomizzation
SEED = 444

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('Number of GPUs being used: ', torch.cuda.device_count())
    print('GPU Type: ', torch.cuda.get_device_name(0))


# In[2]:


# Defining adaptive tanh activation function
class AdaptiveTanh(nn.Module):
    def __init__(self):
        super(AdaptiveTanh, self).__init__()
        self.alpha = nn.Parameter(torch.rand(1).to(device))

    def forward(self,val):
        # Apply adaptive scaling to input before passing it to tanh
        scaled_input = self.alpha * val
        return torch.tanh(scaled_input)

# Neural Network definitions
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation='gelu'):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        # Create a list of hidden layers based on user-defined sizes
        self.hidden_layers = nn.ModuleList()
        prev_size = input_size  # Initialize the input size
        for size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(prev_size, size))
            prev_size = size

        if activation == 'tanh':
            self.act_fun = nn.Tanh()
        else:
            self.act_fun = nn.GELU()

        self.out_act_fun = nn.Tanh()
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)



    def forward(self, X):
        for layer in self.hidden_layers:
            X = self.act_fun(layer(X))
        #X = self.output_layer(X)
        X = self.output_layer(X)
        return X


class Combined(nn.Module):            
    def __init__(self, input_size, hidden_sizes, output_size, activation):
        super(Combined, self).__init__()
        self.model1 = NeuralNetwork(input_size, hidden_sizes, output_size, activation)
        self.model2 = NeuralNetwork(input_size, hidden_sizes, output_size, activation)
        #self.n = output_size
        self.act=AdaptiveTanh()
    

    def forward(self, x, y):
        model1_output = self.model1(x)
        model2_output = self.model2(y)
        
        u = torch.matmul(model1_output, model2_output.T)
        u_scaled = self.act(u)
        return u_scaled


# In[3]:


##############################################################
# Auxillary Functions
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)


def hvp_fwdfwd(f, primals, tangents, return_primals=False):
    g = lambda primals: ft.jvp(f, (primals,), (tangents,))[1]
    primals_out, tangents_out = ft.jvp(g, (primals,), (tangents,))
    if return_primals:
        return primals_out, tangents_out
    else:
        return tangents_out


# In[4]:


##############################################################
# Loss Functions

def spinn_loss(apply_fn, ad_fn, tau, train_data, train_data_ic):
    
    def residual_loss(x,t):
        # calculate u
        u = apply_fn(x,t)
        # tangent vector dx/dx
        # assumes t, x, y have same shape (very important)
        v = torch.ones(x.shape)
        # 2nd derivatives of u
        ux,uxx = hvp_fwdfwd(lambda x: ad_fn(x,t), x, v,return_primals=True)
        ut = ft.jvp(lambda t: ad_fn(x,t), (t,), (v,))[1]
        #return torch.mean((ut-0.0001*uxx+5*(u**3-u))**2) - 1e-6*(torch.mean(torch.log10(u**2 + ux**2 + uxx**2)) + torch.mean(torch.log10((u-1)**2 + ux**2 + uxx**2))  + torch.mean(torch.log10((u+1)**2 + ux**2 + uxx**2)))
        return torch.mean((ut-0.0001*uxx+5*(u**3-u))**2) 
    
    
    def initial_loss(x,t,u):
        return torch.mean((apply_fn(x,t) - u)**2)
    

    def moving_loss(x,y,u,tau,h):
        return torch.sum((h**2)*(apply_fn(x,y) - u)**2)/(2*tau)
    

    def boundary_loss(x,y):
        
        loss_u = 0
        loss_ux = 0
        loss_uy = 0
        for i in range(2):
            loss_u += torch.mean((apply_fn(x[i],y[i]) - apply_fn(x[i+2],y[i+2]))**2)
            
            # v_x = torch.ones(x[i].shape)
            # v_y = torch.ones(y[i].shape)
            # ux_lb =  ft.jvp(lambda x: ad_fn(x,y[i]), (x[i],), (v_x,))[1]
            # ux_ub =  ft.jvp(lambda x: ad_fn(x,y[i+2]), (x[i+2],), (v_x,))[1]
            
            # uy_lb =  ft.jvp(lambda y: ad_fn(x[i],y), (y[i],), (v_y,))[1]
            # uy_ub =  ft.jvp(lambda y: ad_fn(x[i+2],y), (y[i+2],), (v_y,))[1]
            
            # loss_ux += torch.mean((ux_lb - ux_ub)**2)
            # loss_uy += torch.mean((uy_lb - uy_ub)**2)
        
        return loss_u + loss_ux + loss_uy

    def energy_loss(x,y,h):
        u = apply_fn(x,y)
        v_x = torch.ones(x.shape).to(device)
        v_y = torch.ones(y.shape).to(device)
        ux =  ft.jvp(lambda x: ad_fn(x,y), (x,), (v_x,))[1]
        uy =  ft.jvp(lambda y: ad_fn(x,y), (y,), (v_y,))[1]
        f = (h**2)*(0.5*(ux**2+uy**2) + 2500*(u**2 - 1)**2)
        #f= h*(0.5*u**2)
        return torch.sum(f)
    
    
    # unpack data
    xd, yd, xg, yg, xb, yb, h = train_data
    xi, yi, ui, _, _ = train_data_ic
    ngpt = xg.size()[0]
    # Computing the loss value
    #res_loss = residual_loss(xc,tc)
    moving_loss = moving_loss(xi, yi, ui, tau, h)
    ener_loss = energy_loss(xg, yg, h)
    bound_loss = boundary_loss(xb,yb)
    loss = ener_loss + moving_loss
    
    return loss.to(device), ener_loss, bound_loss, moving_loss
 

def icgl_loss(apply_fn, train_data_icgl):
    x, y, u, _, _ = train_data_icgl
    loss = 0.5*torch.mean((apply_fn(x,y) - u)**2) + 0.5*torch.mean(torch.abs(apply_fn(x,y) - u))
    return loss.to(device)


# In[5]:


##############################################################
# Training Data Generation

def spinn_train_generator_AC2D(nc,dom):
    
    # Domain Points
    xd = torch.linspace(dom[0], dom[1], nc+1)
    yd = xd
    
    # Gauss points
    xleft = xd[:-1]
    xright = xd[1:]
    xg = torch.zeros(((2*nc),1))
    xg1 = -(1/3**0.5)*(xright - xleft)/2 + (xright + xleft)/2
    xg2 = (1/3**0.5)*(xright - xleft)/2 + (xright + xleft)/2
    xg[0:-1:2,0] = xg1
    xg[1::2,0] = xg2   
    yg = xg
    
    # Boundary points
    temp = dom[0] + (dom[1]-dom[0])*torch.rand((nc, 1))
    xb = [dom[1]*torch.ones(1,1), temp, -dom[0]*torch.ones(1,1), temp]
    yb = [temp, dom[1]*torch.ones(1,1), temp, -dom[0]*torch.ones(1,1)] 
    
    h = (xd[2] - xd[1])/2
    
    return xd, yd, xg, yg, xb, yb, h

def icgl_train_generator_AC2D(nc,dom):
    
    # Domain Points
    xd = torch.linspace(dom[0], dom[1], nc)
        
    # Gauss points
    xleft = xd[:-1]
    xright = xd[1:]
    xg = torch.zeros(((2*(nc-1)),1))
    xg1 = -(1/3**0.5)*(xright - xleft)/2 + (xright + xleft)/2
    xg2 = (1/3**0.5)*(xright - xleft)/2 + (xright + xleft)/2
    xg[0:-1:2,0] = xg1
    xg[1::2,0] = xg2   
    yg = xg
    
    # Initial points
    xi = xg
    yi = yg
   
    # Generating random initial condition 
    # uxi = (xi**2)*torch.cos(torch.pi*xi)*torch.exp(-xi**2)
    # uyi  = (yi**2)*torch.cos(torch.pi*yi)*torch.exp(-yi**2)
    # uxi = torch.sin(torch.pi*xi)
    # uyi = torch.cos(torch.pi*yi)
    # ui = torch.matmul(uxi, uyi.T)
    
    xgmesh_np, ygmesh_np = np.meshgrid(xg.detach().numpy(),yg.detach().numpy(), indexing='ij')
    xgmesh = torch.Tensor(xgmesh_np)
    ygmesh = torch.Tensor(ygmesh_np)
    
    # ui = torch.exp(-xgmesh**2)*torch.cos(torch.pi*xgmesh)*torch.exp(-ygmesh**2)*torch.cos(torch.pi*ygmesh)*(xgmesh**2)*(ygmesh**2)
    theta = torch.zeros(np.shape(xgmesh))
    for i in range(np.size(xgmesh_np,0)):
        for j in range(np.size(xgmesh_np,1)):        
            if xgmesh[i,j] > 0.5:
                theta[i,j] = torch.arctan((ygmesh[i,j] - 0.5)/(xgmesh[i,j] - 0.5))
            elif xgmesh[i,j] == 0.5:
                theta[i,j] = torch.pi + torch.pi/2
            else:
                theta[i,j] = torch.pi + torch.arctan((ygmesh[i,j] - 0.5)/(xgmesh[i,j] - 0.5))
    
    
    num = 0.25 + 0.1*torch.cos(7*theta) - torch.sqrt((xgmesh - 0.5)**2 + (ygmesh - 0.5)**2)
    den = np.sqrt(2)*0.01
    ui = torch.tanh(num/den)
                
    return xi, yi, ui, xgmesh, ygmesh


# In[6]:


##############################################################
# Optimization Steps
def train_step(loss_fn,optimizer,epoch, lossVal, sol_list, tau, train_data_gauss, train_data_icgl):
    # clear the gradients
    optimizer.zero_grad()

    # Losstorch.jit.script(torch.jit.script(
    loss_spinn, ener_loss, bound_loss, moving_loss = loss_fn(spinn, spinn, tau, train_data_gauss, train_data_icgl)
    loss_value = loss_spinn.detach().cpu().numpy()

    if epoch%100 == 0:
        print('Energy Loss:',ener_loss.detach().cpu().numpy(),', Bound Loss:',bound_loss.detach().cpu().numpy(),', Moving Loss:',moving_loss.detach().cpu().numpy(),', Total Loss:',loss_value, ', iter:', epoch)

    loss_spinn.backward()

    # Update model weights
    optimizer.step()

    return loss_spinn

def train_step_icgl(loss_fn,optimizer,epoch,lossVal_icgl,train_data_icgl):
    # clear the gradients
    optimizer.zero_grad()

    # Loss
    loss_ic = loss_fn(spinn, train_data_icgl)
    loss_value = loss_ic.detach().cpu().numpy()

    if epoch%1000 == 0:
        print(' Total Loss:',loss_value, ', iter:', epoch)

    if epoch%100 == 0:
        lossVal_icgl.append(loss_value)

    loss_ic.backward()

    # Update model weights
    optimizer.step()

    return loss_ic

def closure():
    # Zero gradients
    lbfgs.zero_grad()

    # Compute loss
    loss, ener_loss,bound_loss,moving_loss = spinn_loss(spinn, spinn, tau, train_data_gauss, train_data_icgl)

    # Backward pass
    loss.backward()

    return loss


# In[7]:


##############################################################
# Defining variables and the network
# random key
g_cpu = torch.Generator()
keys =  [g_cpu.manual_seed(SEED),g_cpu.manual_seed(SEED),g_cpu.manual_seed(SEED)]


# dataset
nc = 1024 # user input
nc_icgl = 512 # user input
filename_icgl = '/home/vmattey/SeparablePINN/data/AC/u0_IC3_N_' + str(nc_icgl)+'.mat'
data_icgl = scipy.io.loadmat(filename_icgl)
dom = [0, 1]

# User Input for Size of Neural Network
input_size = 1  # You can change this to the desired number of input features
hidden_sizes = [128, 128, 128, 128]  # You can specify the number of neurons in each hidden layer
output_size = 256

# Epochs for training
epochs_icgl = 20001
epochs_spinn = 2001
# epochs_pinn_init = 1001
lbfgs_epochs = 31

# Time Stepping
dt = 1E-5
nsteps = 300 # Number of time steps to run
ratio = 10 # Scaling factor

# NN Activation
activation = 'gelu' # Choose either tanh or gelu

# Grid for saving and predicting the solution
N = 512 # Number of Elements in each spatial direction
xgrid = torch.linspace(dom[0], dom[1], N+1).to(device)
ygrid = torch.linspace(dom[0], dom[1], N+1).to(device)
t = 0

train_data_gauss = spinn_train_generator_AC2D(nc,dom)
train_data_icgl = icgl_train_generator_AC2D(nc,dom)

# Create an instance of the neural network
spinn = Combined(input_size,hidden_sizes,output_size, activation).to(device)


# Define an optimizer
adam = optim.Adam(spinn.parameters(),lr=0.001)  # You can adjust the learning rate (lr) as needed
scheduler = lr_scheduler.LinearLR(adam,start_factor=1,end_factor=0.1,total_iters=epochs_spinn)
#scheduler_init = lr_scheduler.LinearLR(adam,start_factor=1,end_factor=0.1,total_iters=epochs_pinn_init)
scheduler_icgl = lr_scheduler.LinearLR(adam,start_factor=1,end_factor=0.1,total_iters=epochs_icgl)
lbfgs = optim.LBFGS(
    spinn.parameters(),
    lr = 1.0,
    max_iter=50000,
    max_eval=50000,
    history_size=50,
    tolerance_grad=1e-5,
    tolerance_change=1.0*np.finfo(float).eps,
    line_search_fn="strong_wolfe",
    )
lossVal = []
lossVal_icgl = []
sol_list = []
upred = []
start = time.time()
# Training with ADAM for Initial Condition
for epoch in range(epochs_icgl):
        start_time = time.time()
        loss_fn = icgl_loss
        train_step_icgl(loss_fn,adam,epoch,lossVal_icgl,train_data_icgl)
        scheduler_icgl.step()

upred.append(spinn(xgrid.reshape(N+1,1),ygrid.reshape(N+1,1)))

for i in range(nsteps):
    if i < 100:
        tau = dt
    else:
        tau = ratio*dt
    xi, yi, ui, xgmesh, ygmesh = train_data_icgl
    ui = spinn(xi, yi)
    ui.detach_()
    train_data_icgl = xi, yi, ui, xgmesh, ygmesh
    for epoch in range(lbfgs_epochs):
            running_loss = 0.0
            # Update weights
            lbfgs.step(closure)
            # Update the running loss
            loss = closure()
            running_loss += loss.item()
            if epoch%10 == 0:
                print(f"Epoch: {epoch + 1:02}/{lbfgs_epochs:02} Loss: {running_loss:.5e}")
                loss_fn = spinn_loss
                loss_spinn, ener_loss, _, moving_loss = loss_fn(spinn, spinn, tau, train_data_gauss, train_data_icgl)
                temp = [loss_spinn.detach().cpu().numpy(), ener_loss.detach().cpu().numpy(),moving_loss.detach().cpu().numpy()]
                lossVal.append(temp)
    
    upred.append(spinn(xgrid.reshape(N+1,1),ygrid.reshape(N+1,1)))
    t += tau
    print('Sim Time: ', t)

print('Total training time: ',time.time()-start)


# ## Model Saving Utilities

# In[10]:


path = '/pscratch/sd/v/vmattey/SPINN_AC_2D/results_data_aux/AC_2D_IC3/N_1024/'

import scipy.io
u_pred = []
for u in upred:
    u_pred.append(u.detach().cpu().numpy())

uu = {'upred':u_pred}
scipy.io.savemat(path+'upred_2D_IC1_noTanh.mat',uu)

loss_array_icgl = np.array(lossVal_icgl)
loss_array = np.array(lossVal)

loss_dict_icgl = {'loss_icgl':loss_array_icgl}
scipy.io.savemat(path+'loss_icgl_noTanh.mat',loss_dict_icgl)

loss_dict = {'loss_spinn':loss_array}
scipy.io.savemat(path+'loss_spinn_noTanh.mat',loss_dict)


# ## Plotting results for Deep Ritz Testing

# In[ ]:


step = 101

xgrid = torch.linspace(-1, 1, N+1).resize(N+1,1)
ygrid = torch.linspace(-1, 1, N+1).resize(N+1,1)
xmesh, ymesh = np.meshgrid(xgrid.detach().numpy(),ygrid.detach().numpy())
ypred = upred[step-1]

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
plt.figure()
plt.pcolor(xmesh, ymesh, ypred.detach().cpu().numpy(), cmap = 'turbo', label ='Predicted')
plt.colorbar()

plt.figure()
surf = ax.plot_surface(xmesh, ymesh, ypred.detach().cpu().numpy(), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.set_xlabel('X')
ax.set_ylabel('Y')

fig.colorbar(surf, shrink=0.5, aspect=5)

