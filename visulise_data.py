import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
import json

class visulise_data:
    def __init__(self):
        df_E = pd.read_csv("Data_files/E_field.csv", header=None)
        df_H = pd.read_csv("Data_files/H_field.csv", header = None)
        plt.style.use("seaborn-v0_8")
        self.df_Dielectric = pd.read_csv("Data_files/Dielectric.csv", header=None)
        self.array_E = df_E.values[:, :-1].astype(float)
        self.dielectric_list = []

        with open('Meta_data\Dielectric.json', 'r') as f:
            data = json.load(f)

        self.dielectric_list = []
        for item in data:
            self.dielectric_list.append({"Permitivity": item['Permitivity'], "Conductivity": item['Conductivity'], "Position": item['Position']})

        self.z_index = np.arange(0,np.shape(self.array_E)[1],1)
        self.array_H = df_H.values[:, :-1].astype(float)

    def plot_frame(self,frame):  
        fig, ax = plt.subplots(1)
        ax.set_ylim(np.min(self.array_H)*1.1,np.max(self.array_E)*1.1)
        ax.set_ylabel("$Amplitude$")
        ax.set_xlabel("$Spatial \ Index$")
        for dielectric in self.dielectric_list:
            permitivity = dielectric['Permitivity']
            conductivity = dielectric['Conductivity']
            position = dielectric['Position']
            if position is not None:
                ax.axvspan(position[0],position[1], alpha=0.3, color='blue', label = "$\epsilon_r={} \\ \sigma ={}$".format(permitivity,conductivity))
        
        ax.plot(self.array_E[frame])

        plt.show()

    def plot_animate(self):
        #self.array_E = np.abs(np.fft.fft(self.array_E))
        fig, ax = plt.subplots(1)
        ax.set_ylim(np.min(self.array_E)*1.1,np.max(self.array_E)*1.1)
        ax.set_ylabel("$Amplitude$")
        ax.set_xlabel("$Spatial \ Index$")
        for dielectric in self.dielectric_list:
            permitivity = dielectric['Permitivity']
            conductivity = dielectric['Conductivity']
            position = dielectric['Position']
            if position is not None:
                ax.axvspan(position[0],position[1], alpha=0.3, color='blue', label = "$\epsilon_r={} \\ \sigma ={}$".format(permitivity,conductivity))
        
        line, = ax.plot(self.array_E[0], label = "Electric field")
        def animate(i):
            line.set_ydata(self.array_E[i])
            return line,    
        ax.legend(loc = "best")
        ani = FuncAnimation(fig, animate, frames=1000, interval=10, blit=True)
        """writer = PillowWriter(fps=15,
                                 metadata=dict(artist='Me'),
                                 bitrate=1800)
        ani.save('EM_water.gif', writer=writer)"""
        plt.show()
    
    def plot_intensity(self, frame):
        fig, axs = plt.subplots(2)

        fig.suptitle('EM Pulse')
        axs[0].plot(np.abs(self.array_E[frame])**2)
        axs[0].plot((0.5/self.df_Dielectric-1)/3, 'k--', linewidth=0.75)
        axs[0].text(170, 0.5, '$\epsilon_r$ = {}'.format(4),
        horizontalalignment='center')
        axs[0].set_ylabel("$E_x$")
        axs[1].plot(np.abs(self.array_H[frame])**2)
        axs[1].set_ylabel("$H_y$")
        axs[2].set_xlabel("Spatial Index along $z$-axis")
        axs[1].plot((0.5/self.df_Dielectric-1)/3, 'k--', linewidth=0.75)
        axs[1].text(170, 0.5, '$\epsilon_r$ = {}'.format(4),
        horizontalalignment='center')
        plt.show()
    
Results = visulise_data()
Results.plot_animate()