import os
from ray.tune.registry import register_env
import time

from flow.core.params import VehicleParams
from flow.core.params import NetParams
from flow.core.params import InitialConfig
from flow.core.params import EnvParams
from flow.core.params import SumoParams
from flow.networks import Network
from flow.utils.registry import make_create_env

from stable_baselines3 import *

from palo_alto_sumo_att_env import PaloAltoSumoAtt
from evaluation import Experiment
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan


if __name__ == "__main__":
    """ Setup flow parameters """
    net_params = NetParams(
        template={
            "net": os.path.join(os.getcwd(), "sumo_CA_car_att/sID_0.net.xml"),
            "vtype": os.path.join(os.getcwd(), "sumo_CA_car_att/dist_config.xml"),
            "rou": os.path.join(os.getcwd(), "sumo_CA_car_att/fringe100.rou.xml")
        }
    )

    new_vehicles = VehicleParams()

    env_params = EnvParams(warmup_steps=15, clip_actions=False)
    initial_config = InitialConfig()

    train = True
    if train:
        sim_params = SumoParams(render=False, sim_step=1, restart_instance=True)
    else:
        sim_params = SumoParams(render=True, sim_step=1, restart_instance=True)

    flow_params = dict(
        exp_tag='template',
        env_name=PaloAltoSumoAtt,
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

    """ Setup model """
    model = PPO("MlpPolicy", env, verbose=1)

    """ Setup ego vehicle """
    ego_veh_model = PPO("MlpPolicy", env, verbose=1)
    ego_veh_model_path = "/home/songanz/docker_home_flow/flow_evaluation/log/stable_baseline_3/2022-07-12_14-18-45/rl_model_9000_steps.zip"
    env.load_ego_vehicle(ego_veh_model, ego_veh_model_path)

    if train:
        log_dir = os.path.join("./log/att_env/stable_baseline_3/", time.strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(log_dir, exist_ok=True)
        # setup callback
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir,
                                                 name_prefix='rl_model')
        # model.learn(total_timesteps=int(1e5), callback=checkpoint_callback)  # debug
        model.learn(total_timesteps=int(1e7), callback=checkpoint_callback)

    # setup experiment
    exp = Experiment(flow_params)

    # setup rl policy
    rl_actions = model.next_action

    # run the sumo simulation
    info_dict = exp.run(1, rl_actions=rl_actions)
    print(info_dict)

