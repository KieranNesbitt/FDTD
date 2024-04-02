import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

class visulise_data:
    def __init__(self):
        df_E = pd.read_csv("E_field.csv", header=None)
        df_H = pd.read_csv("H_field.csv", header = None)
        self.df_Dielectric = pd.read_csv("Dielectric.csv", header=None)

        self.array_E = df_E.values[:, :-1].astype(float)
        self.array_H = df_H.values[:, :-1].astype(float)

    def plot(self,frame):  
        fig, axs = plt.subplots(2)
        """axs[0].set_ylim(self.array_E.min()*1.1, self.array_E.max()*1.1)
        axs[1].set_ylim(self.array_H.min()*1.1, self.array_H.max()*1.1)"""

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
        axs[1].plot((0.5/self.df_Dielectric-1)/3, 'k--', linewidth=0.75)
        axs[1].text(170, 0.5, '$\epsilon_r$ = {}'.format(4),
        horizontalalignment='center')
        plt.show()
    
Results = visulise_data()
Results.plot_intensity(500)