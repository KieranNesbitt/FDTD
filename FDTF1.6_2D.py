import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functools import wraps
import time
import os
from alive_progress import alive_bar
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
                cell_spacing: np.float32 = 0.001,
                Dimensions: int = 2,
                courant_number: float = 0.5,
                 ):
        self.mu_0 = 1.25663706e-6
        self.eps_0 = 8.85418782e-12
        self.cell_spacing = cell_spacing
        self.courant_number = courant_number
        self.delta_t: np.float32 = self.cell_spacing/6e8
        
        self.N_x,self.N_y = shape
        #Field arrays
        ## Will be used to store the calc fields
        self.H_field_x = np.zeros((self.N_x, self.N_y), dtype=np.float32)
        self.H_field_y = np.zeros((self.N_x, self.N_y), dtype=np.float32)
        self.E_field_x = np.zeros((self.N_x, self.N_y), dtype=np.float32)
        self.E_flux_x = np.zeros((self.N_x, self.N_y), dtype=np.float32)

        #Lists will be appened at each time step
        self.E_field_list =[]
        self.H_field_list_x =[]
        self.H_field_list_y =[]

        #Material parameters
        self.rel_eps = np.ones((self.N_x, self.N_y), dtype= np.float32)
        self.rel_mu = np.ones((self.N_x, self.N_y), dtype=np.float32)
        self.conductivity = np.zeros((self.N_x, self.N_y), dtype= np.float32)
        self.dielectric_list = []

        open(f"{os.getcwd()}\Meta_data\Dielectric.json", 'w')
        meta_data = {"Cell_spacing": self.cell_spacing, "Time_Delta": self.delta_t, "Dimensions": Dimensions, "Shape": (self.N_x, self.N_y)}
        write_json(meta_data, f"{os.getcwd()}\Meta_data/Grid.json", "w")
        write_csv(self.rel_eps, f"{os.getcwd()}\Data_files\Dielectric.csv")

    def set_source(self, source, position: tuple, Amplitude): #Set the source parameters, using a class function as it allows for variables to set beforehand
        self.source = source
        self.position_source = position
        metadata = {"Position": position, "Amplitude": Amplitude}
        write_json(metadata, f"{os.getcwd()}\Meta_data\Source_2D.json", "w")

    def update_source(self): #Updates the electric field at self.position for each tick
        self.E_flux_x[self.position_source[0][0]:self.position_source[0][1],self.position_source[1][0]:self.position_source[1][1]] += self.source(self.time_step)
        #self.E_flux_x[250,250] += self.source(self.time_step)

    def append_to_list(self):
        self.E_field_list.append(np.copy(self.E_field_x))
        self.H_field_list_x.append(np.copy(self.H_field_x))   
        self.H_field_list_y.append(np.copy(self.H_field_y))   

    def update_E_flux(self):
        self.E_flux_x[1:self.N_x, 1:self.N_y] += 0.5 * (
            self.H_field_y[1:self.N_x, 1:self.N_y] - self.H_field_y[:self.N_x-1, 1:self.N_y] -
            self.H_field_x[1:self.N_x, 1:self.N_y] + self.H_field_x[1:self.N_x, :self.N_y-1]
        )

    def update_E(self):
        self.E_field_x[1:self.N_x, 1:self.N_y] = self.alpha[1:self.N_x, 1:self.N_y] * self.E_flux_x[1:self.N_x, 1:self.N_y]

    def update_H(self):
        self.H_field_x[:-1, :-1] += 0.5 *self.rel_mu[:-1, :-1]* (
            self.E_field_x[:-1, :-1] - self.E_field_x[:-1, 1:] 
        )
        self.H_field_y[:-1, :-1] += 0.5 **self.rel_mu[:-1, :-1]* (
            self.E_field_x[1:, :-1] - self.E_field_x[:-1, :-1]
        )

    def define_constants(self):
        self.alpha = 1/(self.rel_eps + (self.conductivity*self.delta_t/self.eps_0))
        self.beta = self.conductivity*(self.delta_t/self.eps_0)

    def add_dieletric(self, mask_Type, Values: tuple):
        eps, mu, conductivity = Values
        mask_Type.create(self.rel_eps, eps)
        mask_Type.create(self.rel_mu, mu)
        mask_Type.create(self.conductivity, conductivity)

        #The following exports meta data and values of the dielectric for plotting
        metadata = {"Permitivity": eps, "Conductivity": conductivity}
        self.dielectric_list.append(metadata)
        df = pd.DataFrame(self.rel_eps)
        df.to_csv(f"{os.getcwd()}\Data_files\Dielectric_2D.csv", index=False, header=None)
        
    @timeit
    def run(self, total_time):
        self.boundary_low = [0, 0]
        self.boundary_high = [0, 0]
        self.define_constants()
        
        with alive_bar(total_time, bar = "filling", spinner = "waves") as bar:
            for self.time_step in np.arange(0,total_time,1):
                self.update_E_flux()
                self.update_source()
                self.update_E()
                self.update_H()
                
                self.append_to_list()
                bar()
        self.E_field_array = np.array(self.E_field_list)
            
        self.output_to_files()
    def save_array(self, array):
        np.save(f"{os.getcwd()}\Data_files\E_field_array.npy", array)

    @timeit  
    def output_to_files(self):
        write_json(self.dielectric_list, f"{os.getcwd()}\Meta_data\Dielectric_2D.json", "a")
        # Save the entire 3D array
        self.save_array(self.E_field_array)
        
class Source:
    def __init__(self,
                freq: np.float32 = 1,
                cell_spacing: np.float32 = 1,
                courant_number: np.float32 = 0.5,
                Amplitude: np.float32 = 1,
                tau: np.float32 = 1,
                Position: tuple= None, 
                 ):
        self.Position = Position
        self.mu_0 = 1.25663706e-6
        self.eps_0 = 8.85418782e-12
        self.freq = freq
        self.cell_spacing = cell_spacing
        self.courant_number = courant_number
        self.delta_t =self.cell_spacing/6e8
        self.Amplitude = Amplitude
        self.tau = tau
        
        #self.delta_t = np.sqrt(self.mu_0*self.eps_0)*self.courant_number*self.cell_spacing
    def plane_wave(self, time_step):
        return 
    def guassian_40(self,time_step): 
        t0 =100
        spread = 30
        return self.Amplitude*np.exp(-0.5*((t0- time_step)/spread)**2)
    
    def Sinusodial(self, time_step):
        if time_step <= self.tau[0]:
            self.Amplitude_ramped = self.ramp_up(time_step)
        #if time_step >= self.tau[1]:
        #    self.Amplitude_ramped = self.ramp_down(time_step)
        return self.Amplitude_ramped*np.sin(2*np.pi*(self.freq)*self.delta_t*time_step)
    
    def ramp_up(self,time_step):
        return self.Amplitude*(time_step/self.tau[0])
    
    def ramp_down(self, time_step):
        return self.Amplitude*(1-time_step/self.tau[1])
        
def main():#In this Simulation E is normalised by eliminating the electric and magnetic constant from both E and H
    ##Done so that the amplitudes match
    import Dielectric_Mask as Mask
    source = Source(cell_spacing=cellspacing, freq=freq_in, tau=(100,1), Amplitude=Amplitude, Position=source_position)
    FDTD = Grid(shape = (Grid_Size), cell_spacing=cellspacing)
    FDTD.set_source(source.Sinusodial, position = source_position, Amplitude=Amplitude)
    #FDTD.add_dieletric(Mask.Square((250,250), 10), (1.5,1.5,0))
    FDTD.add_dieletric(Mask.Ellipsoid(a=100,b=100, r_x=50, r_y=10), Values=(2,2,0))
    FDTD.run(time_max)

if __name__ == "__main__":
    Grid_Size = (201,201)
    freq_in = 400e9
    wavelength = 3e8/freq_in
    source_position = ((100,101),(100,101))
    cellspacing = wavelength/50 #Cellspacing is determined by the number of cells per wavelength, Standard is 50 ##Looks the best 
    time_max = 1500
    Amplitude = 10
    main()

#Old code
"""def update_H(self):
        delta_E: np.float32 = self.E_field[1:] - self.E_field[:-1]
        self.H_field[:-1] += self.gamma[:-1] * delta_E

    def update_E(self):
        delta_E = self.H_field[1:]-self.H_field[:-1]
        self.E_field[1:] = self.E_field[1:]*self.alpha[1:]+(self.beta[1:]/self.Delta_z)*(delta_E)"""