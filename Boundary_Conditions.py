import numpy as np
class PML:
    def __init__(self, Grid,PML_thickness, dx:float):
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
        self.dx = dx
        self.dt = dx/6e8

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
    
    def PML_create(self, conductivityE: float,conductivityH: float,PML_gradient = 3):
        self.alpha_x,self.alpha_y = self.PML_creation(self.alpha_x,self.alpha_y, self.PML_thickness)
        self.alpha_mx,self.alpha_my = self.PML_creation(self.alpha_mx,self.alpha_my, self.PML_thickness)
        self.sigma_x, self.sigma_y = self.PML_sigma_creation(self.sigma_x, self.sigma_y,self.PML_thickness, conductivityE, PML_gradient)
        self.sigma_mx,self.sigma_my = self.PML_sigma_creation(self.sigma_mx,self.sigma_my,self.PML_thickness, conductivityH, PML_gradient)

        bex = np.exp(-(self.sigma_x/(self.k + self.alpha_x))*(self.dt/self.eps_0))
        bey = np.exp(-(self.sigma_y/(self.k + self.alpha_y))*(self.dt/self.eps_0))
        bhx = np.exp(-(self.sigma_mx/(self.k + self.alpha_mx))*(self.dt/self.eps_0))
        bhy = np.exp(-(self.sigma_my/(self.k + self.alpha_my))*(self.dt/self.eps_0))
        b_constants = (bex,bey, bhx, bhy)
        aex = (bex -1 )*(conductivityE/(conductivityE+self.alpha_x))
        aey = (bey - 1 )*(conductivityE/(conductivityE+self.alpha_y))
        ahx = (bhx - 1 )*(conductivityH/(conductivityH+self.alpha_mx))
        ahy =  (bhy - 1 )*(conductivityH/(conductivityH+self.alpha_my))
        a_constants = (aex,aey,ahx,ahy)
        return a_constants, b_constants