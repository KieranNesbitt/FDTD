import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functools import wraps
import time

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
                impedance_0: np.float16 = 377.0,
                rel_permitivity: np.float16 = 1.0,
                rel_permibility: np.float16 = 1.0,
                total_time: int = 250
                 ):
        self.impedance_0, self.Courant_number =impedance_0, Courant_number
        self.rel_permitivity, self.rel_permibility = rel_permitivity, rel_permibility
        self.total_time = total_time
        self.N_x,self.N_y = shape
        self.H_field = np.zeros(self.N_x, dtype=np.float16)
        self.E_field = np.zeros(self.N_x, dtype=np.float16)
        self.E_field_list =[]
        self.H_field_list =[]

    
    def append_to_list(self):

    # Append the E_field to the list
        self.E_field_list.append(np.copy(self.E_field))
        
        self.H_field_list.append(np.copy(self.H_field))

    def set_boundary_conditions(self):
        self.boundary_low = [0, 0]
        self.boundary_high = [0, 0]        

    def boundary_conditions(self):
        self.E_field[0] = self.boundary_low.pop(0)
        self.boundary_low.append(self.E_field[1])
        self.E_field[self.N_x - 1] = self.boundary_high.pop(0)
        self.boundary_high.append(self.E_field[self.N_x - 2])

    def set_source(self, source):
        self.source = source

    def update_source(self):
        self.E_field[0] += self.source(self.time_step)

    def update_H(self, index):
        self.H_field[index] = self.H_field[index] + self.Courant_number*(self.E_field[index + 1] - self.E_field[index])
    
    def update_E(self, index):
        self.E_field[index] = self.E_field[index] + self.Courant_number*(self.H_field[index] - self.H_field[index-1])

    def run(self, total_time):
        self.time = np.arange(0,total_time+1, 1)
        self.m_index = np.arange(0,self.N_x-1, 1)
        self.set_boundary_conditions()
        
        for self.time_step in self.time:
            
            for index in self.m_index:
                self.update_H(index)
            self.boundary_conditions()

            for index in self.m_index:
                self.update_E(index)
            self.update_source()
            self.append_to_list()

        self.E_field_array = np.array(self.E_field_list)
        self.H_field_array = np.array(self.H_field_list)
        df = pd.DataFrame(self.E_field_array)
        df.to_csv('E_field.csv', index=False, header=None)

        df = pd.DataFrame(self.H_field_array)
        df.to_csv('H_field.csv', index=False, header=None)

def guassian(time_step): 
    t0 =40
    spread = 12 
    return np.exp(-0.5 * ((t0 - time_step) / spread) ** 2)

def sinusoidal(time_step):
    freq_in =700e6
    ddx = 0.01 # Cell size
    dt = ddx / 6e8 # Time step size
    return np. sin(2 * np.pi * freq_in * dt * time_step)
if __name__ == "__main__":
    total_time = 250
    fdtd = Grid(shape = (100,None), Courant_number=0.50)
    fdtd.set_source(guassian)
    fdtd.run(total_time)
