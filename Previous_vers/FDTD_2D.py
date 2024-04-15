import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functools import wraps
import time
import scipy.signal as signal
import multiprocessing as mp
import json

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

class Grid:
    def __init__(self,
                shape: None,
                cell_spacing: np.float16 = 0.001,
                Normalised_E_field: bool = False,
                wavelength: np.float16 = 1,
                pml_thickness: int = 20,  
                pml_sigma_max: float = 10,  
                 ):     
        self.Delta_z = cell_spacing
        self.Delta_t = self.Delta_z/(2*3e8)
        self.N_x,self.N_y = shape
        self.H_field = np.zeros((self.N_x,self.N_y), dtype=np.float64)
        self.E_field = np.zeros((self.N_x,self.N_y), dtype=np.float64)
        self.E_field_list =[]
        self.H_field_list =[]
        #Material parameters
        self.rel_eps = np.ones((self.N_x,self.N_y), dtype= np.float64)
        self.rel_mu = np.ones((self.N_x,self.N_y), dtype=np.float64)
        self.conductivity = np.zeros((self.N_x,self.N_y), dtype= np.float64)
        
    def set_source(self, source, position=None, active_time: float = 1):
        self.source_active_time = active_time
        self.source_active = True
        self.source = source
        if position:
            x, y = position
            self.position = (x, y)  # Include all y values
        else:
            self.position = None

    def update_source(self):
        if self.source_active:
            
            self.E_field[self.position] = self.source(self.time_step)
        if self.source_active_time <= self.time_step:
            self.source_active = False

    def append_to_list(self):
        self.E_field_list.append(np.copy(self.E_field))
        
        self.H_field_list.append(np.copy(self.H_field))       

    def boundary_conditions(self):
        self.E_field[0,:] = self.E_field[1,:]
        self.E_field[-1,:] = self.E_field[-2,:]

        self.E_field[:,0] = self.E_field[:,1]
        self.E_field[:,-1] = self.E_field[:,-2]
        

    def update_H(self):
        delta_E_x = self.E_field[1:, :] - self.E_field[:-1, :]  # Calculate delta_E along the x-axis
        delta_E_y = self.E_field[:, 1:] - self.E_field[:, :-1]  # Calculate delta_E along the y-axis
        self.H_field[:-1, :] += self.gamma[:-1, :] * delta_E_x
        self.H_field[:, :-1] += self.gamma[:, :-1] * delta_E_y

    def update_E(self):
        delta_H_x = self.H_field[1:, :] - self.H_field[:-1, :]  # Calculate delta_H along the x-axis
        delta_H_y = self.H_field[:, 1:] - self.H_field[:, :-1]  # Calculate delta_H along the y-axis
        self.E_field[1:, :] = self.E_field[1:, :] * self.alpha[1:, :] + \
                               (self.beta[1:, :] / self.Delta_z) * delta_H_x
        self.E_field[:, 1:] = self.E_field[:, 1:] * self.alpha[:, 1:] + \
                               (self.beta[:, 1:] / self.Delta_z) * delta_H_y

    def add_dieletric(self, pos: tuple=None, eps: float=1, conductivity: float = 0):
        if pos is not None:
            self.rel_eps[pos[0]:pos[1]] = eps
            self.conductivity[pos[0]:pos[1]] = conductivity 
        metadata = {"Permitivity": eps, "Conductivity": conductivity, "Position": pos}
        with open('Dielectric.json', 'w') as convert_file: 
            convert_file.write(json.dumps(metadata))
            
        df = pd.DataFrame(self.rel_eps)
        df.to_csv('Dielectric.csv', index=False, header=None)

    def define_constants(self):
        self.alpha = ((1-self.Delta_t*self.conductivity*(2*self.rel_eps*8.5418782e-12)**-1)/
                    (1+self.Delta_t*self.conductivity*(2*self.rel_eps*85418782e-12)**-1))
        self.beta = ((self.Delta_t*(self.rel_eps*8.85418782e-12)**-1)/
                    (1+self.Delta_t*self.conductivity*(2*self.rel_eps*85418782e-12)**-1))
        self.gamma = (self.Delta_t)/(self.Delta_z*self.rel_mu*1.25663706e-6)

    @timeit
    def run(self, total_time):
        self.define_constants()
        for self.time_step in np.arange(0,total_time,1):
            self.boundary_conditions()
            self.update_H()
            self.update_E()
            self.update_source()
            self.append_to_list()
        

    def visulise(self):
        self.E_field_array = np.array(self.E_field_list)
        fig, axs = plt.subplots()
        
        i = 150-1
        print(self.E_field_array[i])
        image = axs.imshow(self.E_field_array[i], cmap='Blues') 
        cbar = fig.colorbar(image)
        axs.set_title(f'Slice {i+1}')  
        plt.show()

    """def output_to_csv(self):
        self.E_field_array = np.array(self.E_field_list)
        self.H_field_array = np.array(self.H_field_list)
        df = pd.DataFrame(self.E_field_array)
        df.to_csv('E_field.csv', index=False, header=None)

        df = pd.DataFrame(self.H_field_array)
        df.to_csv('H_field.csv', index=False, header=None)"""

class Source:
    def __init__(self,
                rel_permitivity: np.float16 = 1.0,
                rel_permibility: np.float16 = 1.0,
                wavelength: np.float16 = 1,
                c: np.float64 = 3e8,
                spread: int = 12,
                t0: int = 40,
                amplitude: float = 1
                
                 ):
        self.rel_permitivity, self.rel_permibility = rel_permitivity, rel_permibility
        self.freq = c/wavelength
        self.c = c
        self.spread = spread
        self.t0 = t0
        self.dx=wavelength/10
        self.dt = self.dx/(2*self.c)
        self.amplitude = amplitude

    def gaussian(self, time_step): 
        spread = self.spread
        t0_x, t0_y = self.t0, self.t0

        x = time_step - t0_x
        y = time_step - t0_y

        gaussian_x = np.exp(-0.5 * ((x / spread) ** 2))
        gaussian_y = np.exp(-0.5 * ((y / spread) ** 2))

        gaussian_2d = np.outer(gaussian_x, gaussian_y)
        return gaussian_2d

    def sinusoidal(self,time_step):
        freq_in =400e6
        dx = 0.01 # Cell size
        dt = dx / 6e8 # Time step size
        return np.sin(2 * np.pi * freq_in * dt * time_step)

def main():
    source = Source()
    FDTD = Grid(shape = (201,201))
    FDTD.set_source(source.gaussian, position = [100,100], active_time = 500)
    #FDTD.add_dieletric(pos = (100,150), eps=1.7, conductivity=0.04)
    FDTD.run(500)
    FDTD.visulise()

if __name__ == "__main__":
    main()
#Old code
