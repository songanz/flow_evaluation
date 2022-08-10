import argparse


def train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env', type=str, help='sumo config folder name in sumo_env/config',
                        default='palo_alto_small')
    parser.add_argument('--warmup_steps', type=int, help='sumo environment warmup steps', default=15)
    parser.add_argument('--horizon', type=int, help='sumo environment horizon steps', default=int(1e8))
    parser.add_argument('-r', '--rl_algo', type=str, help='only support options from stable_baselines3',
                        default='SAC')
    parser.add_argument('--total_timesteps', type=int, help='total timesteps for training RL agent',
                        default=int(1e7))
    parser.add_argument('--saved_freq', type=int, help='checkpoint save frequency during training RL agent',
                        default=int(50))
    parser.add_argument('--eval_freq', type=int, help='evaluation frequency during training RL agent',
                        default=50)
    parser.add_argument('-l', '--log', type=str, help='log dir for checkpoints and evaluation results',
                        default='./log')
    parser.add_argument('--ego_rl_algo', type=str, help='for attacker env, load the ego vehicle policy. '
                                                        'only support options from stable_baselines3',
                        default='SAC')
    parser.add_argument('--ego_model_path', type=str, help='for attacker env, load the ego vehicle policy. '
                                                           'ego vehicle saved model path',
                        default='')

    return parser


def load_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env', type=str, help='sumo config folder name in sumo_env/config',
                        default='palo_alto_small')
    parser.add_argument('--num_runs', type=int, help='number of experiments', default=1)
    parser.add_argument('--warmup_steps', type=int, help='sumo environment warmup steps', default=15)
    parser.add_argument('--horizon', type=int, help='sumo environment horizon steps', default=int(1e8))
    parser.add_argument('-r', '--rl_algo', type=str, help='only support options from stable_baselines3',
                        default='SAC')
    parser.add_argument('--model_path', type=str, help='trained model path', default='')
    parser.add_argument('-l', '--log', type=str, help='log dir for checkpoints and evaluation results',
                        default='./log')
    parser.add_argument('--ego_rl_algo', type=str, help='for attacker env, load the ego vehicle policy. '
                                                        'only support options from stable_baselines3',
                        default='SAC')
    parser.add_argument('--ego_model_path', type=str, help='for attacker env, load the ego vehicle policy. '
                                                           'ego vehicle saved model path',
                        default='')
    parser.add_argument('--render', action='store_true', default=True)

    return parser


def plot_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_paths', type=str, help='evaluations.npz files absolut path', nargs='+',
                        default=[])
    return parser
