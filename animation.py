import os
import sys

import SumoNetVis
import matplotlib.pyplot as plt
import matplotlib.animation as animation

if __name__ == '__main__':
    net_path = os.path.join(os.getcwd(), "sumo_env/config/", 'palo_alto_small', "sID_0.net.xml")
    net = SumoNetVis.Net(net_path)
    trajectories = SumoNetVis.Trajectories('/home/songanz/flow_evaluation/log/stable_baseline_3/palo_alto_small/SAC/eval/emission/2022-08-23_11-40-19/fcd-output.xml')

    fig, ax = plt.subplots()
    a = animation.FuncAnimation(fig, trajectories.plot_points, frames=trajectories.timestep_range(), repeat=False,
                                interval=100 * trajectories.timestep, fargs=(net, ax,), blit=True)
    plt.show()
