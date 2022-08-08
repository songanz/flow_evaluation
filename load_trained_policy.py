import os
from ray.tune.registry import register_env
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

from palo_alto_sumo_env import PaloAltoSumo
from evaluation import Experiment


if __name__ == "__main__":
    """ Setup flow parameters """
    net_params = NetParams(
        template={
            "net": os.path.join(os.getcwd(), "sumo_env_config/sumo_CA_car/sID_0.net.xml"),
            "vtype": os.path.join(os.getcwd(), "sumo_env_config/sumo_CA_car/dist_config.xml"),
            "rou": os.path.join(os.getcwd(), "sumo_env_config/sumo_CA_car/fringe100.rou.xml")
        }
    )

    new_vehicles = VehicleParams()

    env_params = EnvParams(warmup_steps=15, clip_actions=False)
    initial_config = InitialConfig()

    sim_params = SumoParams(render=True, no_step_log=True, sim_step=1, restart_instance=True)

    flow_params = dict(
        exp_tag='template',
        env_name=PaloAltoSumo,
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
    env = create_env()

    algo_name = "SAC"
    model_ = getattr(stable_baselines3, algo_name)
    model = model_("MlpPolicy", env, verbose=1)
    model_path = '/home/songanz/flow_evaluation/log/stable_baseline_3/SAC/2022-08-02_21-21-00/model/best_model'
    model.load(model_path)

    # setup experiment
    exp = Experiment(flow_params)

    # setup rl policy
    rl_actions = model.predict

    # run the sumo simulation
    info_dict = exp.run(1, rl_actions=rl_actions)
    print(info_dict)
