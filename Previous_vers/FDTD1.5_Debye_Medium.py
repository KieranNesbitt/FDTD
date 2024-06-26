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

def write_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, header=None)

def write_json(data, filename, mode= "w"):
    with open(filename, mode) as convert_file: 
        convert_file.write(json.dumps(data, indent=4))

class Grid:
    def __init__(self,
                shape: tuple = None,
                cell_spacing: np.float64 = None,
                Dimensions: int = 1,
                courant_number: float = 0.5,
                 ):
        
        self.mu_0 = 1.25663706e-6
        self.eps_0 = 8.85418782e-12
        self.cell_spacing = cell_spacing
        self.courant_number = courant_number
        self.delta_t: np.float64 = self.courant_number*self.cell_spacing/3e8
        self.N_x,self.N_y = shape
        #Field arrays
        ## Will be used to store the calc fields
        self.H_field = np.zeros(self.N_x, dtype=np.float64)
        self.E_field = np.zeros(self.N_x, dtype=np.float64)
        self.E_flux = np.zeros(self.N_x, dtype=np.float64)
        self.I = np.zeros(self.N_x, dtype=np.float64)
        self.S = np.zeros(self.N_x, dtype=np.float64)
        #Lists will be appened at each time step
        self.E_field_list =[]
        self.H_field_list =[]

        #Material parameters
        self.rel_eps = np.ones(self.N_x, dtype= np.float64)
        self.rel_mu = np.ones(self.N_x, dtype=np.float64)
        self.chi1 = np.zeros(self.N_x, dtype=np.float64)
        self.tau = np.ones(self.N_x, dtype=np.float64)
        self.conductivity = np.zeros(self.N_x, dtype= np.float64)
        self.dielectric_list = []

        open('Meta_data/Dielectric.json', 'w')
            
        
    def set_source(self, source, position = None): #Set the soruce parameters, using a class function as it allows for variables to set beforehand
        self.source = source
        self.position_source = position
        metadata = {"Position": position, "Type": str(self.source.__name__)}
        write_json(metadata, 'Meta_data/Source.json', "w")

    def update_source(self): #Updates the electric field at self.position for each tick
        self.E_flux[self.position_source] += self.source(self.time_step)

    def boundary_conditions(self):#Will update to allow for switching between different conditions but for now a simple ABC will be used
        self.E_field[0] = self.boundary_low.pop(0)
        self.boundary_low.append(self.E_field[1])
        self.E_field[self.N_x-1] = self.boundary_high.pop(0)
        self.boundary_high.append(self.E_field[self.N_x - 2])

    def append_to_list(self):
        self.E_field_list.append(np.copy(self.E_field))
        self.H_field_list.append(np.copy(self.H_field))   

    def update_E_flux(self):
        delta_H = self.H_field[:-1] - self.H_field[1:]
        self.E_flux[1:] += self.courant_number * delta_H 

    def update_E(self):
        for i in np.arange(0,self.N_x-1):
            self.E_field[i] = self.alpha[i] * (self.E_flux[i] - self.I[i] - self.del_exp[i]*self.S[i])
            self.I[i] = self.I[i] + self.beta[i]*self.E_field[i]
            self.S[i] = self.del_exp[i]*self.S[i] + self.gamma[i]*self.E_field[i]

    def update_H(self):
        delta_E=self.E_field[:-1]-self.E_field[1:]
        self.H_field[:-1]+=self.courant_number*delta_E/ self.rel_mu[:-1]
 
    def define_constants(self):
        #self.loss = self.delta_t*self.conductivity/(2*self.eps_0*self.rel_eps)
        self.alpha = 1/(self.rel_eps + (self.conductivity*self.delta_t/self.eps_0) + self.chi1*self.delta_t*self.tau)
        self.beta = self.conductivity*(self.delta_t/self.eps_0)
        self.gamma = self.chi1*self.delta_t/self.tau
        self.del_exp = np.array(np.exp(-self.delta_t/self.tau))

    def add_dieletric(self, pos: tuple=None, eps: np.float16 = 1, conductivity: float = 0, mu: np.float16 = 1, chi1: np.float64 = 0, tau: np.float64 = 1):
        if pos is not None:#Catch term if pos is not specified 
            self.rel_eps[pos[0]:pos[1]] = eps
            self.rel_mu[pos[0]:pos[1]] = mu
            self.conductivity[pos[0]:pos[1]] = conductivity
            self.chi1[pos[0]:pos[1]] = chi1
            self.tau[pos[0]:pos[1]] = tau        
        #The following exports meta data and values of the dielectric for plotting
        metadata = {"Permitivity": eps, "Conductivity": conductivity, "Position": pos, "Permibility":mu, "Chi":chi1}
        self.dielectric_list.append(metadata)
        df = pd.DataFrame(self.rel_eps)
        df.to_csv('Data_files/Dielectric.csv', index=False, header=None)
        
    @timeit
    def run(self, total_time): 
        self.boundary_low = [0, 0]
        self.boundary_high = [0, 0]
        self.define_constants()
        for self.time_step in np.arange(0,total_time):
            self.update_E_flux()
            self.update_source()
            self.update_E()
            self.boundary_conditions()
            self.update_H()
            self.append_to_list()
            
        self.output_to_csv()

    @timeit  
    def output_to_csv(self):#Mainly to allow for plotting the data whithout needing to re-run the simulation
        self.E_field_array = np.array(self.E_field_list)
        self.H_field_array = np.array(self.H_field_list)
        np.savetxt('Data_files/E_field.csv', self.E_field_array, delimiter=',')
        np.savetxt('Data_files/H_field.csv', self.H_field_array, delimiter=',')
        write_json(self.dielectric_list, 'Meta_data/Dielectric.json', "a")
        
class Source:
    def __init__(self,
                freq: np.float64 = 1,
                cell_spacing: np.float64 = 1,
                courant_number: np.float64 = 0.5,
                Amplitude: np.float64 = 1,
                tau: np.float64 = 1,
                Ramp_Up: bool = False,
                Ramp_Down: bool = False
                 ):
        self.mu_0 = 1.25663706e-6
        self.eps_0 = 8.85418782e-12
        self.freq = freq
        self.cell_spacing = cell_spacing
        self.courant_number = courant_number
        self.delta_t =self.cell_spacing/6e8
        self.Amplitude = Amplitude
        self.tau = tau
        Ramp_dic = {"Ramp_Up": self.ramp_up, "Ramp_Down": self.ramp_down}
        #self.delta_t = np.sqrt(self.mu_0*self.eps_0)*self.courant_number*self.cell_spacing

    def Guassian_40(self,time_step): 
        t0 = 40
        spread = 12
        return self.Amplitude*np.exp(-0.5*((t0- time_step)/spread)**2)

    def Sinusodial(self, time_step):
        if time_step <= self.tau:
            self.Amplitude_ramped = self.ramp_up(time_step)
        return self.Amplitude_ramped*np.sin(2*np.pi*(self.freq)*self.delta_t*time_step)
    
    def ramp_up(self,time_step):
        return self.Amplitude*(time_step/self.tau)
    
    def ramp_down(self, time_step):
        return self.Amplitude*(1-time_step/self.tau)
    
    
def main():#In this Simulation E is normalised by eliminating the electric and magnetic constant from both E and H
    ##Done so that the amplitudes match

    source = Source(cell_spacing = cellspacing, freq=freq_in, tau=100, Amplitude=1)
    FDTD = Grid(shape = shape, cell_spacing=cellspacing)
    FDTD.set_source(source.Guassian_40, position = source_position)
    FDTD.add_dieletric((150,175), 1.7,0,1.7,0,1)
    FDTD.run(time_max)

if __name__ == "__main__":
    shape = (201,None)
    freq_in = 400e6
    source_position: int = 100
    cellspacing = 0.1
    time_max = 1001
    main()
#Old code
"""def update_H(self):
        delta_E: np.float64 = self.E_field[1:] - self.E_field[:-1]
        self.H_field[:-1] += self.gamma[:-1] * delta_E

    def update_E(self):
        delta_E = self.H_field[1:]-self.H_field[:-1]
        self.E_field[1:] = self.E_field[1:]*self.alpha[1:]+(self.beta[1:]/self.Delta_z)*(delta_E)"""