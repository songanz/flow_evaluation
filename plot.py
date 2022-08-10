import matplotlib.pyplot as plt
import numpy as np
from utils.parser import plot_parse


if __name__ == '__main__':
    parser = plot_parse()
    args = parser.parse_args()
    fig, ax = plt.subplots()
    x = np.arange(0, 5, 1)

    for p in args.eval_paths:
        data = np.load(p)
        algo_name = p.split('/')[-4]
        kernel_size = 10
        kernel = np.ones(kernel_size) / kernel_size

        x = np.arange(0, len(data['results'])*50, 50)
        y = np.mean(data['results'], 1)
        y = np.convolve(y, kernel, mode='same')
        ci = np.std(data['results'], axis=1)
        ci = np.convolve(ci, kernel, mode='same')

        plot_l = ax.plot(x, y, label=algo_name)
        ax.fill_between(x, (y - ci), (y + ci), alpha=.3)

    ax.set_xlim(15000, 20000)
    ax.plot(x, np.zeros(x.shape), '--')
    ax.set_ylim(-50, 20)
    ax.legend()
    ax.set_xlabel('Total Timesteps')
    ax.set_ylabel('Mean Cumulative Reward (over 5 evaluation)')
    plt.show()
