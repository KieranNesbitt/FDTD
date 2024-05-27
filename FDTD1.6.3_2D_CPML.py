import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functools import wraps
import time
import os
from alive_progress import alive_bar
import json
Path = os.getcwd()

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
                
                shape: tuple,
                cell_spacing: np.float32,
                courant_number: float = 0.5,
                Dimensions: int = 2,
                 ):
        self.mu_0 = 1.25663706e-6
        self.eps_0 = 8.85418782e-12
        self.cell_spacing = cell_spacing
        self.courant_number = courant_number
        self.delta_t: np.float32 = self.cell_spacing/6e8
        
        self.N_x,self.N_y = shape
        self.Grid = np.zeros((self.N_x, self.N_y), dtype=np.float32)
        
        open(f'{Path}\Meta_data\Dielectric_2D.json', 'w')
        meta_data = {"Cell_spacing": self.cell_spacing, "Time_Delta": self.delta_t, "Dimensions": Dimensions, "Shape": (self.N_x, self.N_y)}
        write_json(meta_data, f"{Path}\Meta_data/Grid.json", "w")

    def setup_fields(self):
        #Field arrays
        ## Will be used to store the calc fields
        self.field_arrays()

        #Lists will be appened at each time step
        self.E_field_list_x =[]
        self.E_field_list_y =[]
        self.E_field_list_z =[]
        self.H_field_list_x =[]
        self.H_field_list_y =[]
        self.H_field_list_z =[]

        #Material parameters
        self.material_arrays()
        self.dielectric_list = []
        write_csv(self.rel_eps, f"{Path}\Data_files\Dielectric_2D.csv")

    def set_source(self, source, position: tuple, Amplitude): #Set the source parameters, using a class function as it allows for variables to set beforehand
        self.source = source
        self.position_source = position
        metadata = {"Position": position, "Amplitude": Amplitude}
        write_json(metadata, f"{Path}\Meta_data\Source_2D.json", "w")

    def set_PML_boundary(self, boundary_type):
        boundary_type.setup_grid(self.Grid)

        self.PML_alpha, self.PML_kappa, self.PML_sigma, self.Grid = boundary_type.create()

    def field_arrays(self):

        self.H_field_x = np.zeros_like(self.Grid)
        self.H_field_y = np.zeros_like(self.Grid)
        self.H_field_z = np.zeros_like(self.Grid)

        self.E_field_x = np.zeros_like(self.Grid)
        self.E_field_y = np.zeros_like(self.Grid)
        self.E_field_z = np.zeros_like(self.Grid)

        self.E_flux_x = np.zeros_like(self.Grid)
        self.E_flux_y = np.zeros_like(self.Grid)
        self.E_flux_z = np.zeros_like(self.Grid)
         
        self.PhiE_x = np.zeros_like(self.Grid)
        self.PhiE_y = np.zeros_like(self.Grid)
        self.PhiE_z = np.zeros_like(self.Grid)

        self.PhiH_x = np.zeros_like(self.Grid)
        self.PhiH_y = np.zeros_like(self.Grid)
        self.PhiH_z = np.zeros_like(self.Grid)

    def material_arrays(self):
        self.rel_eps = np.ones_like(self.Grid)
        self.rel_mu = np.ones_like(self.Grid)
        self.conductivity = np.zeros_like(self.Grid)

    def update_source(self): #Updates the electric field at self.position for each tick
        self.E_flux_z[self.position_source[0],self.position_source[1]] += self.source(self.time_step)

    def append_to_list(self):
        self.E_field_list_z.append(np.copy(self.E_field_z))
        self.H_field_list_x.append(np.copy(self.H_field_x))   
        self.H_field_list_y.append(np.copy(self.H_field_y))  
     
    def update_E_flux(self):
        curl_H_y = (self.H_field_y[1:,1:] - self.H_field_y[0:-1,1:])/self.cell_spacing
        curl_H_x = (self.H_field_x[1:,1:] - self.H_field_x[1:,0:-1])/self.cell_spacing
        
        
        self.E_flux_z[1:,1:] += (curl_H_y-curl_H_x)*self.beta_E[1:,1:]

    def update_E_field(self):
        self.E_field_z[1:,1:] = self.alpha_E[1:,1:]*self.E_flux_z[1:,1:]  

    def update_H_field(self):
        curl_E_zx = (self.E_field_z[:,:-1] - self.E_field_z[:, 1:])/self.cell_spacing
        self.H_field_x[:,:-1] *=self.alpha_H[:,:-1]
        self.H_field_x[:,:-1] += self.beta_H[:,:-1]*curl_E_zx

        curl_E_zy = (self.E_field_z[1:,:-1] - self.E_field_z[:-1, :-1])/self.cell_spacing
        self.H_field_y[:-1,:-1] *= self.alpha_H[:-1,:-1]
        self.H_field_y[:-1,:-1] += self.beta_H[:-1,:-1]*curl_E_zy

    def define_constants(self):
        loss_E = (self.conductivity*self.delta_t)/(2*self.rel_eps*self.eps_0)
        loss_H = (self.conductivity*self.delta_t)/(2*self.rel_mu*self.mu_0)
        self.alpha_E = (1 - loss_E)/(1+loss_E)
        self.alpha_H = (1-loss_H)/(1+loss_H)
        self.beta_E = (self.delta_t/(2*self.eps_0*self.rel_eps))/(1+loss_E)
        self.beta_H = (self.delta_t/(2*self.mu_0*self.rel_mu))/(1+loss_H)

    def add_object(self, mask_Type, Values: tuple):
        eps, mu, conductivity = Values
        mask_Type.create(self.rel_eps, eps)
        mask_Type.create(self.rel_mu, mu)
        mask_Type.create(self.conductivity, conductivity)

        #The following exports meta data and values of the dielectric for plotting
        metadata = {"Permitivity": eps, "Permibility": mu,"Conductivity": conductivity}
        self.dielectric_list.append(metadata)
        df = pd.DataFrame(self.rel_eps)
        df.to_csv(f"{Path}\Data_files\Dielectric_2D.csv", index=False, header=None)
    
    @timeit
    def run(self, total_time):
        time_array = np.arange(0,total_time)
        self.define_constants()
        
        with alive_bar(total_time, bar = "filling", spinner = "waves") as bar:
            for self.time_step in time_array:
                self.update_E_flux()
                self.update_source()
                self.update_E_field()
                self.update_H_field()
                self.append_to_list()
                bar()
        self.E_field_array = np.array(self.E_field_list_z)
            
        self.output_to_files()
    def save_array(self, array, path):
        print(f"Saving {path} to file {Path+path}")
        np.save(f"{Path}{path}", array)


    @timeit  
    def output_to_files(self):
        write_json(self.dielectric_list, f"{Path}\Meta_data\Dielectric_2D.json", "a")
        # Save the entire 3D array
        self.save_array(self.E_field_array,"\Data_files\E_field_z_array.npy")
        self.save_array(self.H_field_list_x, "\Data_files\H_field_x_array.npy")
        self.save_array(self.H_field_list_y, "\Data_files\H_field_y_array.npy")
        
class Source:
    def __init__(self,
                freq: np.float32,
                cell_spacing: np.float32,
                Position: tuple,
                courant_number: np.float32 = 0.5,
                Amplitude: np.float32 = 1,
                tau: np.float32 = 1,
                 ):
        self.Position = Position
        self.mu_0 = 1.25663706e-6
        self.eps_0 = 8.85418782e-12
        self.freq = freq
        self.cell_spacing = cell_spacing
        self.courant_number = courant_number
        self.delta_t = self.cell_spacing/6e8
        self.Amplitude = Amplitude
        self.tau = tau
        
    def plane_wave(self, time_step):
        return 
    def guassian_40(self,time_step): 
        t0 =40
        spread = 12
        return self.Amplitude*np.exp(-0.5*((t0 - time_step)/spread)**2)
    
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
    from Boundary_Conditions import CPML
    source = Source(cell_spacing=cellspacing, freq=freq_in, tau=(100,1), Amplitude=Amplitude, Position=source_position)
    FDTD = Grid(shape = (Grid_Size), cell_spacing=cellspacing)
    boundary = CPML(PML_Thickness,cellspacing)
    FDTD.set_PML_boundary(boundary)
    FDTD.setup_fields()
    FDTD.set_source(source.Sinusodial, position = source_position, Amplitude=Amplitude)
    FDTD.run(time_max)

if __name__ == "__main__":
    Grid_Size = (100,100) # Reminder that this is (Rows, Columns)
    freq_in = 400e6
    wavelength = 3e8/freq_in #Used for determing cell spacing
    cellspacing = wavelength/40 #Cellspacing is determined by the number of cells per wavelength, Standard for this sim is 0 ##Looks the best 
    time_max: int = 1001 #Max time step taken
    Amplitude: int = 1
    PML_Thickness = 10
    source_position = (60,60)
    main()
