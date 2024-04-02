import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

class visulise_data:
    def __init__(self, frame=None):
        df_E = pd.read_csv("E_field.csv", header=None)
        df_H = pd.read_csv("H_field.csv", header = None)

        array_2d_E = df_E.values[:, :-1].astype(float)
        array_2d_H = df_H.values[:, :-1].astype(float)

        fig, axs = plt.subplots(2)
        axs[0].set_ylim(array_2d_E.min()*1.1, array_2d_E.max()*1.1)
        axs[1].set_ylim(array_2d_H.min()*1.1, array_2d_H.max()*1.1)

        fig.suptitle('EM Pulse')
        axs[0].plot(array_2d_E[frame])
        axs[0].set_ylabel("$E_x$")
        axs[1].plot(array_2d_H[frame])
        axs[1].set_ylabel("$H_y$")
        
       
        plt.show()

visulise_data(200)