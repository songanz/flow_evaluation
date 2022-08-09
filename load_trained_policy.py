import os
from ray.tune.registry import register_env
import sys
stable_baselines3_path = os.path.join(os.getcwd(), "stable-baselines3")
sys.path.append(stable_baselines3_path)

import json

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
from utils.evaluation import Experiment
from utils.parser import load_parser


if __name__ == "__main__":
    parser = load_parser()
    args = parser.parse_args()

    env_dir = args.env
    num_runs = args.num_runs
    warmup_steps = args.warmup_steps
    horizon = args.horizon
    rl_algo = args.rl_algo
    model_path = args.model_path
    log = args.log
    render = args.render

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

    sim_params = SumoParams(render=render, no_step_log=True, sim_step=1, restart_instance=True)

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

    if env_dir == 'palo_alto_with_attacker' and args.ego_model_path != '':
        """ Setup ego vehicle """
        ego_veh_model_ = getattr(stable_baselines3, args.ego_rl_algo)
        ego_veh_model = ego_veh_model_("MlpPolicy", env, verbose=1)
        ego_veh_model_path = args.ego_model_path
        env.load_ego_vehicle(ego_veh_model_path, ego_veh_model)
    elif env_dir == 'palo_alto_with_attacker' and args.ego_model_path == '':
        print("Need trained ego vehicle model")
        sys.exit()

    model_ = getattr(stable_baselines3, rl_algo)
    model = model_("MlpPolicy", env, verbose=1)
    model.load(model_path)

    # setup experiment
    exp = Experiment(flow_params, env=env)

    # setup rl policy
    rl_actions = model.predict

    # run the sumo simulation
    info_dict = exp.run(num_runs, rl_actions=rl_actions)
    print(info_dict)

    # save info dict
    base_folder = os.path.join(log, "stable_baseline_3/", env_dir, rl_algo)
    eval_path = os.path.join(base_folder, 'eval/')
    eval_file_path = os.path.join(eval_path, 'eval.json')
    os.makedirs(eval_path, exist_ok=True)
    json_f = json.dumps(info_dict)
    with open(eval_file_path, 'w') as f:
        f.write(json_f)
