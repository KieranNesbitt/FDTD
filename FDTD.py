import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functools import wraps
import time
import scipy.signal as signal
import multiprocessing as mp

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
                Courant_number: np.float16 = None,  
                rel_permitivity: np.float16 = 1.0,
                rel_permibility: np.float16 = 1.0,
                total_time: int = 250,
                cell_spacing: np.float16 = 0.01,
                Normalised_E_field: bool = False,
                conductivity: np.float16 = 1
                 ):
        self.rel_permitivity, self.rel_permibility = rel_permitivity, rel_permibility
        self.total_time = total_time
        self.impedance_0 = 1
        self.conductivity = conductivity
        if not Normalised_E_field:
            self.impedance_0 = 377
        self.Delta_z = cell_spacing
        self.Delta_t = self.Delta_z/(2*3e8)
        self.Courant_number = 3e8*(self.Delta_t/self.Delta_z)
        self.N_x,self.N_y = shape
        self.H_field = np.zeros(self.N_x, dtype=np.float64)
        self.E_field = np.zeros(self.N_x, dtype=np.float64)
        self.rel_eps = np.ones(self.N_x)
        self.rel_mu = np.ones(self.N_x)
        self.loss_array = np.ones(self.N_x)
        self.E_field_list =[]
        self.H_field_list =[]

    
    def append_to_list(self):

    # Append the E_field to the list
        self.E_field_list.append(np.copy(self.E_field))
        
        self.H_field_list.append(np.copy(self.H_field))

    def set_boundary_conditions(self, Low_BC= None, High_BC= None, pulse_time=None):
        self.pulse_time = pulse_time
        self.boundary_low = [0, 0]
        self.boundary_high = [0, 0]        

    def create_dielectric(self, pos = None):
        if pos == None:
            pos = [0,self.N_x]

        self.rel_eps[pos[0]:pos[1]] /= self.rel_permitivity
        df = pd.DataFrame(self.rel_eps)
        df.to_csv('Dielectric.csv', index=False, header=None)
    
    def create_diamagentic(self, pos = None):
        if pos == None:
            pos = [0, self.N_x]
        self.rel_mu[pos[0]:pos[1]] /=self.rel_permibility
        df = pd.DataFrame(self.rel_mu)
        df.to_csv('Diamagnetic.csv', index=False, header=None)
        
    def create_lossy_medium(self, pos = None):
        if pos == None:
            pos = [0, self.N_x]
        loss = (0.01 / 6e8)*self.conductivity/(2*self.rel_permitivity*8.854e-12)
        self.loss_array[pos[0]:pos[1]] = (1 - loss)/(1 + loss)
        self.rel_eps[pos[0]:pos[1]] = 0.5*(self.rel_permitivity*(1+loss))

    def boundary_conditions(self):
        """if self.pulse_time != None:
            if self.time_step >= self.pulse_time:"""
        """self.E_field[0] = self.boundary_low.pop(0)
        self.boundary_low.append(self.E_field[1])"""
        self.E_field[self.N_x - 1] = self.boundary_high.pop(0)
        self.boundary_high.append(self.E_field[self.N_x - 2])

    def set_source(self, source, position):
        self.source = source
        self.position = position

    def update_source(self):
        self.E_field[self.position] += self.source(self.time_step)
    
    def update_H(self):
        for index in self.m_index:
            self.H_field[index] = self.H_field[index] + self.Courant_number*(self.E_field[index + 1] - self.E_field[index])/self.impedance_0/self.rel_mu[index]
    
    def update_E(self):
        for index in self.m_index:
            self.E_field[index] = self.E_field[index]*self.loss_array[index] + (self.Courant_number*self.rel_eps[index])*(self.H_field[index] - self.H_field[index-1])*self.impedance_0
    @timeit
    def run(self, total_time):
        self.time = np.arange(0,total_time+1, 1)
        self.m_index = np.arange(0,self.N_x-1, 1)
        self.set_boundary_conditions()
        
        for self.time_step in self.time:
            self.update_E()
            self.update_source()
            self.boundary_conditions()
            self.update_H()
            self.append_to_list()
            

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
                freq: np.float16 = 1,
                c: np.float64 = 3e8,
                spread: int = 1,
                t0: int = 0,
                 ):
        self.rel_permitivity, self.rel_permibility = rel_permitivity, rel_permibility
        self.freq = freq
        self.c = c
        self.spread = spread
        self.t0 = t0
    def guassian(self,time_step): 
        t0 =40
        spread = 12 
        return np.exp(-0.5 * ((t0 - time_step) / spread) ** 2)

    def Guassian_pulse(self,time_step):
        t0=40
        f0 = 100e6 #Hz
        return np.exp(-(time_step-t0)**2/(2*time_step**2)*np.cos(2*np.pi*f0*(time_step-t0)))

    def sinusoidal(self,time_step):
        freq_in =400e6
        dx = 0.01 # Cell size
        dt = dx / 6e8 # Time step size
        return np.sin(2 * np.pi * freq_in * dt * time_step)

@timeit
def main():

    total_time = 1000
    source = Source(rel_permitivity=4, freq = 400e6)
    fdtd = Grid(shape = (401,None), rel_permitivity=4 , Normalised_E_field=True, conductivity= 0.04)
    fdtd.create_lossy_medium([100,200])
    fdtd.set_source(source.sinusoidal, 0)
    fdtd.run(total_time)
if __name__ == "__main__":
    main()