import os
from utils.parser import animation_parser

import SumoNetVis
import matplotlib.pyplot as plt
import matplotlib.animation as animation

if __name__ == '__main__':
    parser = animation_parser()
    args = parser.parse_args()
    env_dir = args.env
    log_path = args.log_path

    net_path = os.path.join(os.getcwd(), "sumo_env/config/", env_dir, "sID_0.net.xml")
    net = SumoNetVis.Net(net_path)
    trajectories = SumoNetVis.Trajectories(log_path)

    fig, ax = plt.subplots()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ani = animation.FuncAnimation(fig, trajectories.plot_points, frames=trajectories.timestep_range(), repeat=False,
                                  interval=300 * trajectories.timestep, fargs=(net, ax,), blit=True)
    # to show
    plt.show()

    # to save as mp4
    # ani.save('animation.mp4', writer="ffmpeg")

    # to save as gif
    # ani.save('animation.gif', writer='imagemagick')
