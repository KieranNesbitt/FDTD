import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

class Grid:
    def __init__(self,
                shape: None,
                Courant_number: np.float16 = None,
                impedance_0: np.float16 = 377.0,
                rel_permitivity: np.float16 = 1.0,
                rel_permibility: np.float16 = 1.0,
                 ):
        self.impedance_0, self.Courant_number =impedance_0, Courant_number
        self.rel_permitivity, self.rel_permibility = rel_permitivity, rel_permibility
        self.N_x = shape
        self.H_field = np.zeros(self.N_x, dtype=np.float16)
        self.E_field = np.zeros(self.N_x, dtype=np.float16)
        self.E_field_df = pd.DataFrame()
        with open("E_field", "w") as self.E_field_output:
            pass
        with open("H_field", "w") as self.H_field_output:
            pass
    
    def output_to_csv(self):
        with open("E_field", "a") as self.E_field_output:
            np.savetxt(self.E_field_output, self.E_field, fmt='%1.3f', newline=", ")
            self.E_field_output.write("\n")
        with open("H_field", "a") as self.H_field_output:
            np.savetxt(self.H_field_output, self.H_field, fmt='%1.3f', newline=", ")
            self.H_field_output.write("\n")


    def set_source(self, source):
        self.source = source

    def update_source(self):
        self.E_field[100] += self.source(self.time_step)

    def update_H(self, index):
        self.H_field[index] = self.H_field[index] + 0.5*(self.E_field[index + 1] - self.E_field[index])
    
    def update_E(self, index):
        self.E_field[index] = self.E_field[index] + 0.5*(self.H_field[index] - self.H_field[index-1])

    def run(self, total_time):
        self.time = np.arange(0,total_time, 1)
        self.m_index = np.arange(0,self.N_x-1, 1)
        
        for self.time_step in self.time:
            
            for index in self.m_index:
                self.update_H(index)
            for index in self.m_index:
                self.update_E(index)
            self.update_source()
            self.output_to_csv()


def guassian(time_step):return np.exp(-(time_step -30)*(time_step -30)/(100))
            
fdtd = Grid(shape = 200, Courant_number=1)
fdtd.set_source(guassian)
fdtd.run(250)
