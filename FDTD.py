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
        self.H_field = np.zeros(self.N_x, dtype=np.float64)
        self.E_field = np.zeros(self.N_x, dtype=np.float64)
        self.E_field_list =[]
        self.H_field_list =[]
        #Material parameters
        self.rel_eps = np.ones(self.N_x, dtype= np.float64)
        self.rel_mu = np.ones(self.N_x, dtype=np.float64)
        self.conductivity = np.zeros(self.N_x, dtype= np.float64)
        
    def set_source(self, source, position = None, active_time: float = 1):
        self.source_active_time = active_time
        self.source_active = True
        self.source = source
        self.position = position

    def update_source(self):
        if self.source_active:
            self.E_field[self.position] = self.source(self.time_step)
        if self.source_active_time <= self.time_step:
            self.source_active = False

    def append_to_list(self):
        self.E_field_list.append(np.copy(self.E_field))
        
        self.H_field_list.append(np.copy(self.H_field))       

    def boundary_conditions(self):
        self.E_field[0] = self.boundary_low.pop(0)
        self.boundary_low.append(self.E_field[1])
        self.E_field[self.N_x-1] = self.boundary_high.pop(0)
        self.boundary_high.append(self.E_field[self.N_x - 2])
    
    def update_H(self):
        delta_E: np.float64 = self.E_field[1:] - self.E_field[:-1]
        self.H_field[:-1] += self.gamma[:-1] * delta_E

    def update_E(self):
        delta_E = self.H_field[1:]-self.H_field[:-1]
        self.E_field[1:]=self.E_field[1:]*self.alpha[1:]+(self.beta[1:]/self.Delta_z)*(delta_E)

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
        self.alpha = (1-self.Delta_t*self.conductivity*(2*self.rel_eps*8.5418782e-12)**-1)/(1+self.Delta_t*self.conductivity*(2*self.rel_eps*85418782e-12)**-1)
        self.beta = (self.Delta_t*(self.rel_eps*8.85418782e-12)**-1)/(1+self.Delta_t*self.conductivity*(2*self.rel_eps*85418782e-12)**-1)
        self.gamma = (self.Delta_t)/(self.Delta_z*self.rel_mu*1.25663706e-6)

    @timeit
    def run(self, total_time):
        self.define_constants()
        self.boundary_low = [0, 0]
        self.boundary_high = [0, 0]
        for self.time_step in np.arange(0,total_time,1):

            self.boundary_conditions()
            self.update_H()
            self.update_E()
            self.update_source()

            self.append_to_list()
        self.output_to_csv()

    @timeit
    def output_to_csv(self):
        self.E_field_array = np.array(self.E_field_list)
        self.H_field_array = np.array(self.H_field_list)
        df = pd.DataFrame(self.E_field_array)
        df.to_csv('E_field.csv', index=False, header=None)

        df = pd.DataFrame(self.H_field_array)
        df.to_csv('H_field.csv', index=False, header=None)

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

    def guassian(self,time_step): 
        return np.exp(-0.5 * ((self.t0-time_step) / self.spread) ** 2)

    """def sinusoidal(self,time_step):
        return self.amplitude*np.sin(2 * np.pi * self.freq * self.dt * time_step)"""
    def sinusoidal(self,time_step):
        freq_in =400e6
        dx = 0.01 # Cell size
        dt = dx / 6e8 # Time step size
        return np.sin(2 * np.pi * freq_in * dt * time_step)

def main():
    source = Source()
    FDTD = Grid(shape = (201,None))
    FDTD.set_source(source.guassian, position = [50], active_time = 90)
    FDTD.add_dieletric(pos = (100,150), eps=1.7, conductivity=0.04)
    FDTD.run(1000)

if __name__ == "__main__":
    main()
#Old code
