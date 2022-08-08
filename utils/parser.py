import argparse


def train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env', type=str, help='sumo config folder name in sumo_env/config',
                        default='palo_alto_small')
    parser.add_argument('--warmup_steps', type=int, help='sumo environment warmup steps', default=15)
    parser.add_argument('-r', '--rl_algo', type=str, help='Only support options from stable_baselines3',
                        default='SAC')
    parser.add_argument('--total_timesteps', type=int, help='total timesteps for training RL agent',
                        default=int(1e7))
    parser.add_argument('--eval_freq', type=int, help='evaluation frequence during training RL agent',
                        default=50)
    parser.add_argument('-l', '--log', type=str, help='log dir for checkpoints and evaluation results',
                        default='./log')
