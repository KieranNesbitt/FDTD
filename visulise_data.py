import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

class visulise_data:
    def __init__(self):
        df_E = pd.read_csv("E_field.csv", header=None)
        df_H = pd.read_csv("H_field.csv", header = None)
        plt.style.use("seaborn-v0_8")
        self.df_Dielectric = pd.read_csv("Dielectric.csv", header=None)

        self.array_E = df_E.values[:, :-1].astype(float)
        self.z_index = np.arange(0,np.shape(self.array_E)[0],1)
        self.array_H = df_H.values[:, :-1].astype(float)

    def plot_frame(self,frame):  
        fig, axs = plt.subplots(3)


        fig.suptitle('EM Pulse')
        axs[0].plot(self.array_E[frame])
        axs[0].plot((0.5/self.df_Dielectric-1)/3, 'k--', linewidth=0.75)
        axs[0].text(170, 0.5, '$\epsilon_r$ = {}'.format(4),
        horizontalalignment='center')
        axs[0].set_ylabel("$E_x$")
        axs[1].plot(self.array_H[frame])
        axs[1].set_ylabel("$H_y$")
        axs[1].plot((0.5/self.df_Dielectric-1)/3, 'k--', linewidth=0.75)
        axs[1].text(170, 0.5, '$\epsilon_r$ = {}'.format(4),
        horizontalalignment='center')

        axs[2].plot(self.array_E[frame]+self.array_H[frame])
        axs[2].set_ylabel("$E_x + H_y$")
        axs[2].set_xlabel("Spatial Index along $z$-axis")
        axs[2].plot((0.5/self.df_Dielectric-1)/3, 'k--', linewidth=0.75)
        axs[2].text(170, 0.5, '$\epsilon_r$ = {}'.format(4),
        horizontalalignment='center')


        plt.show()

    def plot_animate(self):
       
        fig, ax = plt.subplot()
        lines = [ax.plot([],[])[0] for _ in self.array_E]



        


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
Results.plot_frame(1000)