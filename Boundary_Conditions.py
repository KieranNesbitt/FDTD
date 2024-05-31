import numpy as np
import matplotlib.pyplot as plt
plt.style.use("bmh")

class CPML:
    def __init__(self,PML_thickness, dw:float):
        self.PML_thickness = PML_thickness
        self.k_max=10
        self.m = 3
        self.r=0.005
        self.alpha_max=0.25
        self.sigma_max = (0.8*(self.m+1))/(377*dw)
        self.m_a = 1
        self.eps_0 = 8.85418782e-12
        self.dw = dw
        self.dt = dw/6e8

    def setup_grid(self, Grid):
        self.Grid = np.pad(Grid, self.PML_thickness, mode="constant", constant_values=0)
        self.Grid_shape = self.Grid.shape
        self.Grid_PML = np.pad(Grid, self.PML_thickness, mode="linear_ramp", end_values=(self.PML_thickness, self.PML_thickness))
        
    def setup_PML_sigma(self,):
        self.PML_sigma_x = np.zeros_like(self.Grid_PML)
        self.PML_sigma_y = np.zeros_like(self.Grid_PML)
        for i in range(self.PML_thickness): 
            self.PML_sigma_y[i,:] = self.sigma_max*((self.PML_thickness-i)/self.PML_thickness)**self.m
            self.PML_sigma_y[-i-1,:] = self.sigma_max*((self.PML_thickness-i)/self.PML_thickness)**self.m
            self.PML_sigma_x[:,i] = self.sigma_max*((self.PML_thickness-i)/self.PML_thickness)**self.m
            self.PML_sigma_x[:,-i-1] = self.sigma_max*((self.PML_thickness-i)/self.PML_thickness)**self.m

    def setup_PML_constants(self):
        self.PML_beta_x = np.exp(-self.PML_sigma_x*(1+self.r)*(self.dt/self.eps_0))
        self.PML_beta_y = np.exp(self.PML_sigma_y*(1+self.r)*(self.dt/self.eps_0))

        self.PML_alpha_x = self.PML_beta_x/(1+self.r)
        self.PML_alpha_y = self.PML_beta_y/(1+self.r)

    def create(self):
        self.setup_PML_sigma()
        self.setup_PML_constants()

        return self.Grid

    def check_constants(self):
        fig, ax = plt.subplots(2,2,layout='constrained')
        ax[0,0].set_title(r"$\beta(x)$")
        ax[0,1].set_title(r"$\alpha(y)$")
        ax[1,0].set_title(r"$\beta(x)$")
        ax[1,1].set_title(r"$\alpha(y)$")
        im1=ax[0,0].imshow(self.PML_beta_x,aspect='auto')
        im2=ax[0,1].imshow(self.PML_alpha_y,aspect='auto')
        plt1 = ax[1,0].scatter(np.arange(0, self.PML_beta_x.shape[0]), self.PML_beta_x[0,:], marker=".",c =self.PML_beta_x[0,:], cmap = "viridis")
        plt2 = ax[1,1].scatter(np.arange(0,self.PML_alpha_y.shape[0]),self.PML_alpha_y[:,0],marker=".", c=self.PML_alpha_y[:,0], cmap="viridis")
        fig.colorbar(im1, ax=ax[:,0])
        fig.colorbar(im2, ax=ax[:,1])

        plt.show()
        
class PML:
    def __init__(self,PML_thickness, graded_conductivity: bool=False):
        self.PML_thickness = PML_thickness
        self.graded_conductivity = graded_conductivity
    
    def setup_grid(self, Grid):
        self.Grid = Grid
    
    def create(self, conductivity: float, PML_Gradient: float = 3):  
        self.Grid_PML = np.pad(self.Grid, self.PML_thickness, mode="linear_ramp", end_values=(self.PML_thickness, self.PML_thickness))
        if self.graded_conductivity:
            conductivity = np.pad(self.Grid, self.PML_thickness, mode="linear_ramp", end_values=(conductivity,conductivity))
        self.PML_1 = conductivity*(self.Grid_PML/self.PML_thickness)**PML_Gradient
        
        self.PML_2 = (1/(1+self.PML_1))
        self.PML_3 = (1-self.PML_1)/(1+self.PML_1)


        return self.PML_1, self.PML_2, self.PML_3, np.pad(self.Grid, self.PML_thickness, mode="constant", constant_values=0)

"""b = CPML(64, 0.01)
b.setup_grid(np.zeros((100,100)))
b.create()
b.check_constants()"""