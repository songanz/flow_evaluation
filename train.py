import os
from ray.tune.registry import register_env
import time
import sys
stable_baselines3_path = os.path.join(os.getcwd(), "stable-baselines3")
sys.path.append(stable_baselines3_path)

import gym

from flow.core.params import VehicleParams
from flow.core.params import NetParams
from flow.core.params import InitialConfig
from flow.core.params import EnvParams
from flow.core.params import SumoParams
from flow.networks import Network
from flow.utils.registry import make_create_env

import stable_baselines3

from sumo_env.palo_alto_sumo_env import PaloAltoSumo
from sumo_env.palo_alto_sumo_att_env import PaloAltoSumoAtt
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from utils.parser import train_parser


class CheckpointCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` calls
    to ``env.step()``.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

    :param save_freq:
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param verbose:
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save(path)
            try:
                self.model.save_replay_buffer(os.path.join(self.save_path, 'replay_buffer.pkl'))
                print('Off policy, saving replay buffer')
            except:
                print('On policy')
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")
        return True


if __name__ == "__main__":
    parser = train_parser()
    args = parser.parse_args()

    env_dir = args.env
    warmup_steps = args.warmup_steps
    horizon = args.horizon
    rl_algo = args.rl_algo
    total_timesteps = args.total_timesteps
    saved_freq = args.saved_freq
    eval_freq = args.eval_freq
    log = args.log

    """ Setup log dirs """
    base_folder = os.path.join(log, "stable_baseline_3/", env_dir, rl_algo)
    log_dir = os.path.join(base_folder, time.strftime('%Y-%m-%d_%H-%M-%S'))
    emission_path = os.path.join(log_dir, 'emission/')
    os.makedirs(emission_path, exist_ok=True)
    model_path = os.path.join(log_dir, 'model/')
    os.makedirs(model_path, exist_ok=True)
    eval_path = os.path.join(log_dir, 'eval/')
    os.makedirs(eval_path, exist_ok=True)

    """ Setup flow parameters """
    net_params = NetParams(
        template={
            "net": os.path.join(os.getcwd(), "sumo_env/config/", env_dir, "sID_0.net.xml"),
            "vtype": os.path.join(os.getcwd(), "sumo_env/config/", env_dir, "dist_config.xml"),
            "rou": os.path.join(os.getcwd(), "sumo_env/config/", env_dir, "fringe100.rou.xml")
        }
    )

    new_vehicles = VehicleParams()

    env_params = EnvParams(warmup_steps=warmup_steps, clip_actions=False)
    initial_config = InitialConfig()

    sim_params = SumoParams(render=False, no_step_log=False, sim_step=1, restart_instance=True,
                            emission_path=emission_path)

    if env_dir == 'palo_alto_with_attacker':
        env_name = PaloAltoSumoAtt
    else:
        env_name = PaloAltoSumo

    flow_params = dict(
        exp_tag='template',
        env_name=env_name,
        network=Network,
        simulator='traci',
        sim=sim_params,
        env=env_params,
        net=net_params,
        veh=new_vehicles,
        initial=initial_config,
    )

    # number of time steps
    flow_params['env'].horizon = int(horizon)
    """ Register as gym env and create env """
    create_env, gym_name = make_create_env(params=flow_params, version=0)
    register_env(gym_name, create_env)
    env = create_env()
    env_eval = Monitor(gym.envs.make(gym_name))
    if env_dir == 'palo_alto_with_attacker' and args.ego_model_path != '':
        """ Setup ego vehicle """
        ego_veh_model_ = getattr(stable_baselines3, args.ego_rl_algo)
        ego_veh_model = ego_veh_model_("MlpPolicy", env, verbose=1)
        ego_veh_model_path = args.ego_model_path
        env.load_ego_vehicle(ego_veh_model_path, ego_veh_model)
    elif env_dir == 'palo_alto_with_attacker' and args.ego_model_path == '':
        print("Need trained ego vehicle model")
        sys.exit()

    """ Setup model """
    model_ = getattr(stable_baselines3, rl_algo)
    model = model_("MlpPolicy", env, verbose=1)

    logger = configure(eval_path, ['stdout', 'csv', 'tensorboard'])
    model.set_logger(logger)

    # setup callback
    checkpoint_callback = CheckpointCallback(save_freq=saved_freq, save_path=model_path,
                                             name_prefix='rl_model')
    eval_callback = EvalCallback(env_eval, best_model_save_path=model_path, log_path=eval_path,
                                 n_eval_episodes=5, eval_freq=eval_freq)
    callback = CallbackList([checkpoint_callback, eval_callback])

    # Start training
    model.learn(total_timesteps=int(total_timesteps), callback=callback)

