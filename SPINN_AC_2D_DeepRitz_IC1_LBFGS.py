# Code for Separable PINN in PyTorch
# Solving 2D Allen Cahn equation  using separable deep ritz

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('GPU: ', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Number of GPUs being used: ', torch.cuda.device_count())
    print('GPU Type: ', torch.cuda.get_device_name(0))

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
        self.act= nn.Tanh()
    

    def forward(self, x, y):
        model1_output = self.model1(x)
        model2_output = self.model2(y)
        
        u = torch.matmul(model1_output, model2_output.T)
        u_scaled = self.act(u)
        return u_scaled
 
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
    
    return loss.to(device), ener_loss, bound_loss, moving_loss
 

def icgl_loss(apply_fn, train_data_icgl):
    x, y, u, _, _ = train_data_icgl
    loss = 0.5*torch.mean((apply_fn(x,y) - u)**2) + 0.5*torch.mean(torch.abs(apply_fn(x,y) - u))
    return loss.to(device)

##############################################################
# Training Data Generation

def spinn_train_generator_AC2D(nc,dom):
    
    # Domain Points
    xd = torch.linspace(dom[0], dom[1], nc+1)
    yd = xd
    
    # Gauss points
    xleft = xd[:-1]
    xright = xd[1:]
    xg = torch.zeros(((2*(nc)),1))
    xg1 = -(1/3**0.5)*(xright - xleft)/2 + (xright + xleft)/2
    xg2 = (1/3**0.5)*(xright - xleft)/2 + (xright + xleft)/2
    xg[0:-1:2,0] = xg1
    xg[1::2,0] = xg2   
    yg = xg
    
    # Boundary points
    temp = dom[0] + (dom[1]-dom[0])*torch.rand((nc, 1)).to(device)
    xb = [dom[1]*torch.ones(1,1).to(device), temp, -dom[0]*torch.ones(1,1).to(device), temp]
    yb = [temp, dom[1]*torch.ones(1,1).to(device), temp, -dom[0]*torch.ones(1,1).to(device)] 
    
    h = (xd[2] - xd[1])/2
    
    return xd.to(device), yd.to(device), xg.to(device), yg.to(device), xb, yb, h.to(device)

def icgl_train_generator_AC2D(nc,dom):
    
    # Domain Points
    xd = torch.linspace(dom[0], dom[1], nc+1)
        
    # Gauss points
    xleft = xd[:-1]
    xright = xd[1:]
    xg = torch.zeros(((2*(nc)),1))
    xg1 = -(1/3**0.5)*(xright - xleft)/2 + (xright + xleft)/2
    xg2 = (1/3**0.5)*(xright - xleft)/2 + (xright + xleft)/2
    xg[0:-1:2,0] = xg1
    xg[1::2,0] = xg2   
    yg = xg
    
    # Initial points
    xi = xg
    yi = yg
   
    # Generating random initial condition 
   
    xgmesh_np, ygmesh_np = np.meshgrid(xg.detach().numpy(),yg.detach().numpy(), indexing='ij')
    xgmesh = torch.Tensor(xgmesh_np)
    ygmesh = torch.Tensor(ygmesh_np)
    ui = 0.02*torch.cos(4*np.pi*xgmesh)*torch.cos(8*np.pi*ygmesh) - 0.02*torch.cos(6*np.pi*xgmesh)*torch.cos(2*np.pi*ygmesh) + \
        0.01*torch.cos(10*np.pi*xgmesh)*torch.cos(4*np.pi*ygmesh) + 0.02*torch.cos(6*np.pi*xgmesh)*torch.cos(6*np.pi*ygmesh)
    
    # ui = torch.exp(-xgmesh**2)*torch.cos(torch.pi*xgmesh)*torch.exp(-ygmesh**2)*torch.cos(torch.pi*ygmesh)*(xgmesh**2)*(ygmesh**2)
                
    return xi.to(device), yi.to(device), ui.to(device), xgmesh.to(device), ygmesh.to(device)

##############################################################
# Optimization Steps
def train_step(loss_fn,optimizer,epoch, lossVal, sol_list, tau, train_data_gauss, train_data_icgl):
    # clear the gradients
    optimizer.zero_grad()
    
    # Losstorch.jit.script(torch.jit.script(
    loss_spinn, ener_loss, bound_loss, moving_loss = loss_fn(spinn, spinn, tau, train_data_gauss, train_data_icgl)
    loss_value = loss_spinn.detach().cpu().numpy()
    
    if epoch%100 == 0:
        print('Energy Loss:',ener_loss.detach().numpy(),', Bound Loss:',bound_loss.detach().numpy(),', Moving Loss:',moving_loss.detach().numpy(),', Total Loss:',loss_value, ', iter:', epoch)
#        temp = [loss_value, ener_loss.detach().cpu().numpy(),bound_loss.detach().cpu().numpy(),moving_loss.detach().cpu().numpy()]
#        lossVal.append(temp)
    
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


##############################################################
# Defining variables and the network
# random key
g_cpu = torch.Generator()
keys =  [g_cpu.manual_seed(SEED),g_cpu.manual_seed(SEED),g_cpu.manual_seed(SEED)]


# dataset
nc = 256 # user input
dom = [0, 1]

# User Input for Size of Neural Network
input_size = 1  # You can change this to the desired number of input features
hidden_sizes = [128, 128, 128, 128]  # You can specify the number of neurons in each hidden layer
output_size = 256

# Epochs for training
epochs_icgl = 20001
epochs_spinn = 1001
# epochs_pinn_init = 1001
lbfgs_epochs = 31

# Time Stepping
tau = 2E-4
nsteps = 1000 # Number of time steps to run

# NN Activation
activation = 'gelu' # Choose either tanh or gelu

# Grid for Solution
N = 512 # Number of Elements in each spatial direction
xgrid = torch.linspace(dom[0], dom[1], N+1).to(device)
ygrid = torch.linspace(dom[0], dom[1], N+1).to(device)
t = 0


train_data_gauss = spinn_train_generator_AC2D(nc,dom)
train_data_icgl = icgl_train_generator_AC2D(nc,dom)

# Create an instance of the neural network
spinn = Combined(input_size,hidden_sizes,output_size, activation).to(device)
spinn = torch.jit.script(spinn)
#spinn.apply(init_weights)


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
    if i == 0:
        loss_old = temp[0]
    else: 
        if temp[0] == loss_old:
            # Modify the code for ADAM to run for 1000 epochs
            for epoch in range(epochs_spinn):
                start_time = time.time()
                loss_fn = spinn_loss
                train_step(loss_fn,adam,epoch, lossVal, sol_list, tau, train_data_gauss, train_data_icgl)
                scheduler.step()
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
            loss_old = temp[0]
        else:
            loss_old = temp[0]
        
            
                
    upred.append(spinn(xgrid.reshape(N+1,1),ygrid.reshape(N+1,1)))     
    t += tau
    print('Sim Time: ', t)

print('Total training time: ',time.time()-start)

#%%

path = '/home/vmattey/research/spinn/results_data_aux/'
os.chdir(path)

import scipy.io
u_pred = []
for u in upred:
    u_pred.append(u.detach().cpu().numpy())

uu = {'upred':u_pred}
scipy.io.savemat('upred_2D_IC1.mat',uu)


loss_array_icgl = np.array(lossVal_icgl)
loss_array = np.array(lossVal)

loss_dict_icgl = {'loss_icgl':loss_array_icgl}
scipy.io.savemat('loss_icgl.mat',loss_dict_icgl)

loss_dict = {'loss_spinn':loss_array}
scipy.io.savemat('loss_spinn.mat',loss_dict)
