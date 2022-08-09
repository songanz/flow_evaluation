import os
from ray.tune.registry import register_env
import time
import sys
stable_baselines3_path = os.path.join(os.getcwd(), "stable-baselines3")
sys.path.append(stable_baselines3_path)

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
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines3.common.logger import configure
from utils.parser import train_parser


if __name__ == "__main__":
    parser = train_parser()
    args = parser.parse_args()

    env_dir = args.env
    warmup_steps = args.warmup_steps
    rl_algo = args.rl_algo
    total_timesteps = args.total_timesteps
    eval_freq = args.eval_freq
    log = args.log

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

    sim_params = SumoParams(render=False, no_step_log=False, sim_step=1, restart_instance=True)

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
    flow_params['env'].horizon = int(1e8)
    """ Register as gym env and create env """
    create_env, gym_name = make_create_env(params=flow_params, version=0)
    register_env(gym_name, create_env)
    # env = create_env()
    env = DummyVecEnv([lambda: create_env()])
    env = VecCheckNan(env, raise_exception=True)

    if env_dir == 'palo_alto_with_attacker' and args.ego_model_path != '':
        """ Setup ego vehicle """
        ego_veh_model_ = getattr(stable_baselines3, args.ego_rl_algo)
        ego_veh_model = ego_veh_model_("MlpPolicy", env, verbose=1)
        ego_veh_model_path = args.ego_model_path
        env.load_ego_vehicle(ego_veh_model, ego_veh_model_path)
    elif env_dir == 'palo_alto_with_attacker' and args.ego_model_path == '':
        print("Need trained ego vehicle model")
        sys.exit()

    """ Setup model """
    model_ = getattr(stable_baselines3, rl_algo)
    model = model_("MlpPolicy", env, verbose=1)

    base_folder = os.path.join(log, "stable_baseline_3/", env_dir, rl_algo)
    log_dir = os.path.join(base_folder, time.strftime('%Y-%m-%d_%H-%M-%S'))
    model_path = os.path.join(log_dir, 'model/')
    os.makedirs(model_path, exist_ok=True)
    eval_path = os.path.join(log_dir, 'eval/')
    os.makedirs(eval_path, exist_ok=True)

    logger = configure(eval_path, ['stdout', 'csv', 'tensorboard'])
    model.set_logger(logger)

    # setup callback
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=model_path,
                                             name_prefix='rl_model')
    eval_callback = EvalCallback(env, best_model_save_path=model_path, log_path=eval_path,
                                 n_eval_episodes=5, eval_freq=50)
    callback = CallbackList([checkpoint_callback, eval_callback])

    # Start training
    model.learn(total_timesteps=int(1e7), callback=callback)

