import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import json

class visulise_data_1D:
    def __init__(self):
        df_E = pd.read_csv(f"{os.getcwd()}\Data_files\E_field.csv", header=None)
        df_H = pd.read_csv(f"{os.getcwd()}\Data_files\H_field.csv", header = None)
        plt.style.use("seaborn-v0_8")
        self.df_Dielectric = pd.read_csv(f"{os.getcwd()}\Data_files/Dielectric.csv", header=None)
        self.array_E = df_E.values[:, :-1].astype(float)
        self.dielectric_list = []

        with open(f'{os.getcwd()}\Meta_data\Dielectric.json', 'r') as f:
            meta_dieletric = json.load(f)
        with open(f"{os.getcwd()}\Meta_data\Grid.json", "r") as f:
            meta_grid = json.load(f)
        self.cell_spacing, self.Time_Delta = meta_grid["Cell_spacing"], meta_grid["Time_Delta"]
        self.Grid_Shape = meta_grid["Shape"]
        self.dielectric_list = []
        for item in meta_dieletric:
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
    
class visulise_data_2D:
    def plot_2D_animate_imshow(self, save_animation: bool = False):
        
        self.E_field_array = np.abs(np.load(f'{os.getcwd()}\Data_files\E_field_array.npy'))
        self.Dielectric_array = np.loadtxt(open(f"{os.getcwd()}\Data_files\Dielectric_2D.csv", "rb"), delimiter=",", skiprows=1)
        fig,ax = plt.subplots()
        
        im1 = plt.imshow(self.Dielectric_array,animated=True, alpha=0.4, cmap = "Greens")
        im1.set_clim(1)
        im = plt.imshow(self.E_field_array[0], animated=True, cmap="Blues")
        #ax.set(xticks=np.arange(0, self.cell_spacing*self.Grid_Shape[0], self.cell_spacing))
        ax.minorticks_on()
        cb = fig.colorbar(im, ax=ax)
        cb2 = fig.colorbar(im1, ax = ax)
        cb2.ax.set_title("$\epsilon_r$")
        ax.set_xlabel("$X$")
        ax.set_ylabel("$Y$")
        cb.ax.set_title("$|\overrightarrow{E_x}(x,y)|$")
        title = ax.text(0.5,0.95, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha="center")
        #im.set_clim(np.min(self.E_field_array), np.max(self.E_field_array))
        def updatefig(i):
            im.set_array(self.E_field_array[i])
            cb.update_normal(im)
            title.set_text(f"Time step: {i}")
            return im,title,im1,
        ani = FuncAnimation(fig, updatefig, interval=1, frames= 1500, blit=True)
        if save_animation != False:
            writer=PillowWriter(fps=30,
                                 metadata=dict(artist='Me'),
                                 bitrate=1800)
            ani.save(f"{os.getcwd()}\Animation_files\EM_wave.gif", writer=writer)
        plt.show()
    
    def plot_2D_surface(self, frame):
        self.E_field_array = np.load(f"{os.getcwd()}\Data_files\E_field_array.npy")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = np.arange(0, np.shape(self.E_field_array[0])[1])
        y = np.arange(0, np.shape(self.E_field_array[0])[0])
        x,y = np.meshgrid(x,y)
        plot = ax.plot_surface(x,y, self.E_field_array[frame], cmap = "viridis",rstride=2,cstride=2, linewidth=1)
        ax.set_xlabel("$X$")
        ax.set_ylabel("$Y$")
        cb = fig.colorbar(plot)
        cb.ax.set_title("$\overrightarrow{E_x}(x,y)$")
        #plt.show()
def main():
    Results = visulise_data_2D()
    Results.plot_2D_animate_imshow()
    plt.show()

if __name__ == "__main__":
    main()