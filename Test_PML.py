import numpy as np
import matplotlib.pyplot as plt
N_x, N_y = (100,100)
Grid = np.zeros((N_x, N_y))
dx = 0.01
dt = dx/6e8
PML_thickness =40
eps_0 = 8.85418782e-12
alpha_x = np.ones_like(Grid)
alpha_y = np.ones_like(Grid)
alpha_mx = np.ones_like(Grid)
alpha_my = np.ones_like(Grid)
sigma_x = np.zeros_like(Grid)
sigma_y = np.zeros_like(Grid)
sigma_mx = np.zeros_like(Grid)
sigma_my = np.zeros_like(Grid)
k=1

def PML_creation(x_array,y_array, PML_thickness):
    for i in range(PML_thickness+1):
        x_array[i,:],x_array[-i-1,:] = i/PML_thickness,i/PML_thickness
        y_array[:,i],y_array[:,-i-1] = i/PML_thickness,i/PML_thickness
    return x_array, y_array

def PML_sigma_creation(x_array,y_array, PML_thickness,value,PML_gradient=1):
    
    for i in range(PML_thickness+1):
        value_i = value*((PML_thickness-i)/PML_thickness)**PML_gradient
        x_array[i,:],x_array[-i-1,:] = value_i, value_i
        y_array[:,i],y_array[:,-i-1] = value_i,value_i
    return x_array, y_array

alpha_x,alpha_y = PML_creation(alpha_x,alpha_y, PML_thickness)
alpha_mx,alpha_my = PML_creation(alpha_mx,alpha_my, PML_thickness)
sigma_x, sigma_y = PML_sigma_creation(sigma_x, sigma_y,PML_thickness, 0.25)
sigma_mx,sigma_my = PML_sigma_creation(sigma_mx,sigma_my,PML_thickness, 0.25, 3)

bex = np.exp(-(sigma_x/(k + alpha_x))*(dt/eps_0))
bey = np.exp(-(sigma_y/(k + alpha_y))*(dt/eps_0))
bhx = np.exp(-(sigma_mx/(k + alpha_mx))*(dt/eps_0))
bhy = np.exp(-(sigma_my/(k + alpha_my))*(dt/eps_0))
b_constants = (bex,bey, bhx, bhy)
aex = (bex -1 )/ dx
aey = (bey - 1 ) / dx
ahx = (bhx - 1 ) / dx
ahy =  (bhy - 1 ) / dx
a_constants = (aex,aey,ahx,ahy)
fig, ax = plt.subplots(2,2)
im=ax[0,0].imshow(bex, cmap="Blues")
ax[0,1].imshow(bey, cmap="Blues")
ax[1,0].imshow(bhx, cmap= "Blues")
ax[1,1].imshow(bhy, cmap="Blues")

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
fig1, ax1 = plt.subplots(1,2)
ax1[0].plot(bex[:,0])
ax1[1].plot(bhx[:,0])
plt.show()