import numpy as np
import matplotlib.pyplot as plt
class CPML:
    def __init__(self,PML_thickness, dw:float):
        self.PML_thickness = PML_thickness
        self.k_max=10
        self.m = 3
        self.alpha_max=0.25
        self.sigma_max = (0.8*(self.m+1))/(377*dw)
        self.m_a = 1
        self.eps_0 = 8.85418782e-12
        self.dw = dw
        self.dt = dw/6e8

    def setup_grid(self, Grid):
        self.Grid = np.pad(Grid, self.PML_thickness, mode="constant", constant_values=0)
        self.Grid_PML = np.pad(Grid, self.PML_thickness, mode="linear_ramp", end_values=(self.PML_thickness, self.PML_thickness))

    def setup_PML_kappa(self):
        self.PML_kappa = 1 + (self.k_max-1)*(self.Grid_PML/self.PML_thickness)**self.m

    def setup_PML_alpha(self,):
        self.PML_alpha = self.alpha_max*((self.Grid_PML)/(self.PML_thickness))**self.m_a
        
    def setup_PML_sigma(self,):
        self.PML_sigma = self.sigma_max*(self.Grid_PML/self.PML_thickness)**self.m

    def create(self):
        self.setup_PML_alpha()
        self.setup_PML_kappa()
        self.setup_PML_sigma()
        return self.PML_alpha, self.PML_kappa, self.PML_sigma, self.Grid 
        
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
