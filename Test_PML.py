import numpy as np
import matplotlib.pyplot as plt
N_x, N_y = (100,100)
Grid = np.zeros((N_x, N_y))
dx = 0.01
dt = dx/6e8
PML_thickness =40
eps_0 = 8.85418782e-12

class PML:
    def __init__(self, Grid,PML_thickness, dw:float):
        self.PML_thickness = PML_thickness
        self.Grid = np.pad(Grid, PML_thickness, mode="constant", constant_values=0)
        self.alpha_x = np.ones_like(self.Grid)
        self.alpha_y = np.ones_like(self.Grid)
        self.alpha_mx = np.ones_like(self.Grid)
        self.alpha_my = np.ones_like(self.Grid)
        self.sigma_x = np.zeros_like(self.Grid)
        self.sigma_y = np.zeros_like(self.Grid)
        self.sigma_mx = np.zeros_like(self.Grid)
        self.sigma_my = np.zeros_like(self.Grid)
        self.k=1
        self.eps_0 = 8.85418782e-12
        self.dw = dw
        self.dt = dw/6e8

    def PML_creation(self,x_array,y_array, PML_thickness):
        for i in range(PML_thickness+1):
            x_array[i,:],x_array[-i-1,:] = i/PML_thickness,i/PML_thickness
            y_array[:,i],y_array[:,-i-1] = i/PML_thickness,i/PML_thickness
        return x_array, y_array

    def PML_sigma_creation(self,x_array,y_array, PML_thickness,value,PML_gradient):
        
        for i in range(PML_thickness+1):
            value_i = value*((PML_thickness-i)/PML_thickness)**PML_gradient
            x_array[i,:],x_array[-i-1,:] = value_i, value_i
            y_array[:,i],y_array[:,-i-1] = value_i,value_i
        return x_array, y_array
    
    def create(self, conductivityE: float,conductivityH: float,PML_gradient = 3):

        self.alpha_x,self.alpha_y = self.PML_creation(self.alpha_x,self.alpha_y, self.PML_thickness)
        self.alpha_mx,self.alpha_my = self.PML_creation(self.alpha_mx,self.alpha_my, self.PML_thickness)
        self.sigma_x, self.sigma_y = self.PML_sigma_creation(self.sigma_x, self.sigma_y,self.PML_thickness, conductivityE, PML_gradient)
        self.sigma_mx,self.sigma_my = self.PML_sigma_creation(self.sigma_mx,self.sigma_my,self.PML_thickness, conductivityH, PML_gradient)

        sigma_values = (self.sigma_x, self.sigma_y, self.sigma_mx, self.sigma_my)
        bex = np.exp(-(self.sigma_x/(self.k + self.alpha_x))*(self.dt/self.eps_0))
        bey = np.exp(-(self.sigma_y/(self.k + self.alpha_y))*(self.dt/self.eps_0))
        bhx = np.exp(-(self.sigma_mx/(self.k + self.alpha_mx))*(self.dt/self.eps_0))
        bhy = np.exp(-(self.sigma_my/(self.k + self.alpha_my))*(self.dt/self.eps_0))

        b_constants = (bex,bey, bhx, bhy)
        aex = (bex -1 )*(conductivityE/(conductivityE+self.alpha_x*self.k))
        aey = (bey - 1 )*(conductivityE/(conductivityE+self.alpha_y*self.k))
        ahx = (bhx - 1 )*(conductivityH/(conductivityH+self.alpha_mx*self.k))
        ahy =  (bhy - 1 )*(conductivityH/(conductivityH+self.alpha_my*self.k))
        a_constants = (aex,aey,ahx,ahy)
        return a_constants, b_constants, sigma_values ,self.Grid
"""    

boundary = PML(Grid,PML_thickness, dw = dx)
a_constants, b_constants, sigma_array, Grid = boundary.create(conductivityE=0.75, conductivityH=0.75)
aex, aey, ahx, ahy = a_constants
bex, bey, bhx, bhy = b_constants
fig, ax = plt.subplots(2,2)
im=ax[0,0].imshow(bex, cmap="viridis")
ax[0,1].imshow(bey, cmap="viridis")
ax[1,0].imshow(ahx, cmap= "viridis")
ax[1,1].imshow(ahy, cmap="viridis")

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cb = fig.colorbar(im, cax=cbar_ax)
fig1, ax1 = plt.subplots(1)
ax1.plot(bex[:,0], label = "B Constants")
ax1.plot(aex[:,0], label = "A Constant")
plt.show()"""
from Boundary_Conditions import PML as P
b = P(np.zeros((100,100)), 40)
PML_1, PML_2, PML_3, Grid=b.create(0.333, 3)
plt.imshow(PML_1, cmap="plasma", vmax=0.01)
plt.show()