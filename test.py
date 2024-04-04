import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class VisualizeData:
    def __init__(self):
        df_E = pd.read_csv("E_field.csv", header=None)
        df_H = pd.read_csv("H_field.csv", header=None)
        self.df_Dielectric = pd.read_csv("Dielectric.csv", header=None)

        self.array_E = df_E.values[:, :-1].astype(float)
        self.array_H = df_H.values[:, :-1].astype(float)

    def plot(self, frame):
        fig, axs = plt.subplots(3)

        fig.suptitle('EM Pulse')
        axs[0].plot(self.array_E[frame])
        axs[0].plot((0.5 / self.df_Dielectric - 1) / 3, 'k--', linewidth=0.75)
        axs[0].text(170, 0.5, '$\epsilon_r$ = {}'.format(4), horizontalalignment='center')
        axs[0].set_ylabel("$E_x$")
        
        axs[1].plot(self.array_H[frame])
        axs[1].set_ylabel("$H_y$")
        axs[1].plot((0.5 / self.df_Dielectric - 1) / 3, 'k--', linewidth=0.75)
        axs[1].text(170, 0.5, '$\epsilon_r$ = {}'.format(4), horizontalalignment='center')

        axs[2].plot(self.array_E[frame] + self.array_H[frame])
        axs[2].set_ylabel("$E_x + H_y$")
        axs[2].set_xlabel("Spatial Index along $z$-axis")
        axs[2].plot((0.5 / self.df_Dielectric - 1) / 3, 'k--', linewidth=0.75)
        axs[2].text(170, 0.5, '$\epsilon_r$ = {}'.format(4), horizontalalignment='center')

        plt.show()

def update(frame):
    visualize_data.plot(frame)

if __name__ == "__main__":
    visualize_data = VisualizeData()
    total_frames = min(len(visualize_data.array_E), len(visualize_data.array_H))
    ani = animation.FuncAnimation(plt.gcf(), update, frames=total_frames, interval=200, repeat=False)
    plt.show()
