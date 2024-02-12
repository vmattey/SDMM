# Code for Separable PINN in PyTorch
# Solving Linear Elasticity problem using separable deep ritz

# Details of the equation
# u_t = lap(u) - (1/eps^2)*(u^2 - 1)*u; 
# x,y = [0,1] x [0,1], tspan = [0 0.022], eps = 0.01
# u, grad(u) periodic on the boundary
# u0 = 0.02cos(4*pi*x)cos(8*pi*y) - 0.02cos(6*pi*x)cos(2*pi*y) + 0.01cos(10*pi*x)cos(4*pi*y) + 0.02cos(6*pi*x)cos(6*pi*y)



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
import os
# Seed for randomizzation
SEED = 444

##############################################################
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

        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    

    def forward(self, X):
        for layer in self.hidden_layers:
            X = self.act_fun(layer(X))
        X = self.output_layer(X)
        return X
    

class Combined(nn.Module):            
    def __init__(self, input_size, hidden_sizes, output_size, activation):
        super(Combined, self).__init__()
        self.model1 = NeuralNetwork(input_size, hidden_sizes, output_size, activation)
        self.model2 = NeuralNetwork(input_size, hidden_sizes, output_size, activation)
        #self.n = output_size
    

    def forward(self, x, y):
        model1_output = self.model1(x)
        model2_output = self.model2(y)
        
        u = torch.matmul(model1_output, model2_output.T)
        return u
 
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
        v_x = torch.ones(x.shape)
        v_y = torch.ones(y.shape)
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
    
    return loss, ener_loss, bound_loss, moving_loss
 

def icgl_loss(apply_fn, train_data_icgl):
    x, y, u, _, _ = train_data_icgl
    loss = torch.mean((apply_fn(x,y) - u)**2)
    return loss

##############################################################
# Training Data Generation

def spinn_train_generator_AC2D(nc,dom):
    
    # Domain Points
    xd = torch.linspace(dom[0], dom[1], nc)
    yd = xd
    
    # Gauss points
    xleft = xd[:-1]
    xright = xd[1:]
    xg = torch.zeros(((2*(nc-1)),1))
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
    '''ui = 10*(0.02*torch.cos(4*np.pi*xgmesh)*torch.cos(8*np.pi*ygmesh) - 0.02*torch.cos(6*np.pi*xgmesh)*torch.cos(2*np.pi*ygmesh) + \
        0.01*torch.cos(10*np.pi*xgmesh)*torch.cos(4*np.pi*ygmesh) + 0.02*torch.cos(6*np.pi*xgmesh)*torch.cos(6*np.pi*ygmesh))'''
    
    ui = torch.exp(-xgmesh**2)*torch.cos(torch.pi*xgmesh)*torch.exp(-ygmesh**2)*torch.cos(torch.pi*ygmesh)*(xgmesh**2)*(ygmesh**2)
                
    return xi, yi, ui, xgmesh, ygmesh

##############################################################
# Optimization Steps
def train_step(loss_fn,optimizer,epoch, lossVal, sol_list, tau, train_data_gauss, train_data_icgl):
    # clear the gradients
    optimizer.zero_grad()
    
    # Losstorch.jit.script(torch.jit.script(
    loss_spinn, ener_loss, bound_loss, moving_loss = loss_fn(spinn, spinn, tau, train_data_gauss, train_data_icgl)
    loss_value = loss_spinn.detach().numpy()
    
    if epoch%100 == 0:
        print('Energy Loss:',ener_loss.detach().numpy(),', Bound Loss:',bound_loss.detach().numpy(),', Moving Loss:',moving_loss.detach().numpy(),', Total Loss:',loss_value, ', iter:', epoch)
        temp = [loss_value, ener_loss.detach().numpy(),bound_loss.detach().numpy(),moving_loss.detach().numpy()]
        lossVal.append(temp)
    
    loss_spinn.backward()
    
    # Update model weights
    optimizer.step()
    
    return loss_spinn

def train_step_icgl(loss_fn,optimizer,epoch,lossVal_icgl,train_data_icgl):
    # clear the gradients
    optimizer.zero_grad()
    
    # Loss
    loss_ic = loss_fn(spinn, train_data_icgl)
    loss_value = loss_ic.detach().numpy()
    
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


##############################################################
# Defining variables and the network
# random key
g_cpu = torch.Generator()
keys =  [g_cpu.manual_seed(SEED),g_cpu.manual_seed(SEED),g_cpu.manual_seed(SEED)]


# dataset
nc = 1024 # user input
dom = [-1, 1]

# User Input for Size of Neural Network
input_size = 1  # You can change this to the desired number of input features
hidden_sizes = [128, 128, 128, 128]  # You can specify the number of neurons in each hidden layer
output_size = 256
epochs_icgl = 3001
epochs_pinn = 2001
epochs_pinn_init = 1001
tau = 2E-6
activation = 'gelu' # Choose either tanh or gelu
N = 512
xgrid = torch.linspace(dom[0], dom[1], N)
ygrid = torch.linspace(dom[0], dom[1], N)
t = 0


train_data_gauss = spinn_train_generator_AC2D(nc,dom)
train_data_icgl = icgl_train_generator_AC2D(nc,dom)

# Create an instance of the neural network
spinn = Combined(input_size,hidden_sizes,output_size, activation)
spinn = torch.jit.script(spinn)
#spinn.apply(init_weights)


# Define an optimizer
adam = optim.Adam(spinn.parameters(),lr=0.001)  # You can adjust the learning rate (lr) as needed
scheduler = lr_scheduler.LinearLR(adam,start_factor=1,end_factor=0.1,total_iters=epochs_pinn)
scheduler_init = lr_scheduler.LinearLR(adam,start_factor=1,end_factor=0.1,total_iters=epochs_pinn_init)
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
        
upred.append(spinn(xgrid.reshape(N,1),ygrid.reshape(N,1)))   
 
# Training with ADAM
# for epoch in range(epochs_pinn_init):
#         start_time = time.time()
#         loss_fn = spinn_loss
#         train_step(loss_fn,adam,epoch, lossVal, sol_list,tau,train_data_gauss, train_data_icgl)
#         scheduler_init.step()
        
# Training with LBFGS

for i in range(500):
    xi, yi, ui, xgmesh, ygmesh = train_data_icgl
    ui = spinn(xi, yi)
    ui.detach_()
    train_data_icgl = xi, yi, ui, xgmesh, ygmesh
    for epoch in range(21):
            running_loss = 0.0
            # Update weights
            lbfgs.step(closure)
            # Update the running loss
            loss = closure()
            running_loss += loss.item()
            if epoch%10 == 0:
                print(f"Epoch: {epoch + 1:02}/21 Loss: {running_loss:.5e}")
                
    upred.append(spinn(xgrid.reshape(N,1),ygrid.reshape(N,1)))     
    t += tau
    print('Sim Time: ', t)

#%%

upred.append(spinn(xgrid.reshape(N,1),ygrid.reshape(N,1)))    
t += tau
print('Sim Time: ', t)

for i in range(100):
    xi, yi, ui, xgmesh, ygmesh = train_data_icgl
    ui = spinn(xi, yi)
    ui.detach_()
    train_data_icgl = xi, yi, ui, xgmesh, ygmesh

    
    # Training with ADAM
    for epoch in range(epochs_pinn):
            start_time = time.time()
            loss_fn = spinn_loss
            train_step(loss_fn,adam,epoch, lossVal, sol_list,tau,train_data_gauss, train_data_icgl)
            #scheduler.step()
    
    
    upred.append(spinn(xgrid.reshape(N,1),ygrid.reshape(N,1)))     
    t += tau
    print('Sim Time: ', t)
    
print('Total time taken: ', time.time()-start)

#%%
# Training with LBFGS
for epoch in range(10000):
        running_loss = 0.0
        # Update weights
        lbfgs.step(closure)
        # Update the running loss
        loss = closure()
        running_loss += loss.item()
        if epoch%100 == 0:
            print(f"Epoch: {epoch + 1:02}/10000 Loss: {running_loss:.5e}")
        if running_loss <= 1e-5:
            exit

#%% Saving the Model
checkpt = {'model_params':spinn.state_dict(),
                    'optimizer':adam.state_dict(),             
                    'net': spinn
                    }
torch.save(checkpt,'AC_2D_IC1.pt')
#%% Plotting results for Deep Ritz Testing
step = 351

N = 512
xgrid = torch.linspace(-1, 1, N).resize(N,1)
ygrid = torch.linspace(-1, 1, N).resize(N,1)
xmesh, ymesh = np.meshgrid(xgrid.detach().numpy(),ygrid.detach().numpy())
ypred = upred[step-1]

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
plt.figure()
plt.pcolor(xmesh, ymesh, ypred.detach().numpy(), cmap = 'coolwarm', label ='Predicted')
plt.colorbar()

plt.figure()
surf = ax.plot_surface(xmesh, ymesh, ypred.detach().numpy(), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.set_xlabel('X')
ax.set_ylabel('Y')

fig.colorbar(surf, shrink=0.5, aspect=5)


#%% Plotting the results

def plot_ac(xtrain,uexact,upred,t_val):
  plt.figure()
  plt.scatter(xtrain,uexact,label='exact')       
  plt.scatter(xtrain,upred.detach().numpy(),label='predicted')
  plt.legend(loc=1) 
  plt.title(f"t = {t_val}")   
  plt.show()
  
def plot_ac_cmap(x,t,uexact,upred):
    tmesh,xmesh = np.meshgrid(t,x)
    upred = upred.detach().numpy()
    plt.figure()
    plt.pcolor(tmesh,xmesh,upred,cmap='coolwarm')
    plt.title('Predicted')
    plt.figure()
    plt.pcolor(tmesh,xmesh,uexact,cmap='coolwarm')
    plt.title('Exact')
    

t_val = 1 #change this variable based on the time snapshot

# Numerical Solution
data = scipy.io.loadmat('AC_R_1.mat')
x = data['x']
t = data['tt']
u = data['uu']
uexact = u[:,int(t_val*200)]


# Neural Network Prediction
xtest = torch.tensor(x.T,dtype=torch.float32)
ttest = t_val*torch.ones(1,1)
upred = spinn(xtest,ttest)    

plot_ac(x,uexact,upred,t_val)

# For 2D surface plot
ttest = torch.tensor(t.T[:int(t_val*200)],dtype=torch.float32)
upred = spinn(xtest,ttest)  
plot_ac_cmap(x,t.T[:int(t_val*200)],u[:,:int(t_val*200)],upred)

#%% Numerical Quadrature Testing
def spinn_train_generator_AC2D(nc,dom):
    
    # Domain Points
    xd = torch.linspace(dom[0], dom[1], nc)
    yd = xd
    
    # Gauss points
    xleft = xd[:-1]
    xright = xd[1:]
    xg = torch.zeros(((2*(nc-1)),1))
    xg1 = -(1/3**0.5)*(xright - xleft)/2 + (xright + xleft)/2
    xg2 = (1/3**0.5)*(xright - xleft)/2 + (xright + xleft)/2
    xg[0:-1:2,0] = xg1
    xg[1::2,0] = xg2   
    yg = xg
    
    # Boundary points
    temp = -1 + 2*torch.rand((nc, 1))
    xb = [1*torch.ones(1,1), temp, -1*torch.ones(1,1), temp]
    yb = [temp, 1*torch.ones(1,1), temp, -1*torch.ones(1,1)] 
    
    h = (xd[2] - xd[1])/2
    
    return xd, yd, xg, yg, xb, yb, h

_, _, xg, yg, _, _, h = spinn_train_generator_AC2D(1024,[-1,1])

xgmesh_np, ygmesh_np = np.meshgrid(xg.detach().numpy(),yg.detach().numpy(), indexing='ij')
xgmesh = torch.Tensor(xgmesh_np)
ygmesh = torch.Tensor(ygmesh_np)

#f = torch.matmul((xg**2),(yg.T**2))
#f = torch.matmul(torch.exp(-xg**2)*torch.cos(9*np.pi*xg),torch.exp(-yg.T**2)*torch.cos(9*np.pi*yg.T))
k = 1
f = 1/(1+torch.exp(-2*k*xgmesh*ygmesh))
g = torch.sum((h**2)*f)
#%% Saving the Solution
try:
    os.makedirs('./results_data_aux/AC_2D_IC2')
except:
    pass

os.chdir('./results_data_aux/AC_2D_IC2')        

import scipy.io
u_pred = []
for u in upred:
    u_pred.append(u.detach().numpy())

uu = {'upred':u_pred}
scipy.io.savemat('upred_2D_IC2.mat',uu)


loss_array_icgl = np.array(lossVal_icgl)
loss_array = np.array(lossVal)

loss_dict_icgl = {'loss_icgl':loss_array_icgl}
scipy.io.savemat('loss_icgl.mat',loss_dict_icgl)

loss_dict = {'loss_spinn':loss_array}
scipy.io.savemat('loss_spinn.mat',loss_dict)





