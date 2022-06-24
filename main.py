import os
from ray.tune.registry import register_env

from flow.core.params import VehicleParams
from flow.core.params import NetParams
from flow.core.params import InitialConfig
from flow.core.params import EnvParams
from flow.core.params import SumoParams
from flow.networks import Network
from flow.utils.registry import make_create_env

from stable_baselines3 import PPO

from palo_alto_sumo_env import PaloAltoSumo
from evaluation import Experiment
from callbacks import SaveOnBestTrainingRewardCallback


if __name__ == "__main__":
    """ Setup flow parameters """
    net_params = NetParams(
        template={
            "net": os.path.join(os.getcwd(), "sumo_CA_car/sID_0.net.xml"),
            "vtype": os.path.join(os.getcwd(), "sumo_CA_car/dist_config.xml"),
            "rou": os.path.join(os.getcwd(), "sumo_CA_car/fringe100.rou.xml")
        }
    )

    new_vehicles = VehicleParams()

    env_params = EnvParams()
    initial_config = InitialConfig()

    train = True
    if train:
        sim_params = SumoParams(render=False, sim_step=1)
    else:
        sim_params = SumoParams(render=True, sim_step=1)

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
    flow_params['env'].horizon = 100000000
    """ Register as gym env and create env """
    create_env, gym_name = make_create_env(params=flow_params, version=0)
    register_env(gym_name, create_env)
    env = create_env()

    """ Setup model """
    model = PPO("MlpPolicy", env, verbose=1)

    if train:
        log_dir = "log/"
        os.makedirs(log_dir, exist_ok=True)
        # setup callback
        callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
        model.learn(total_timesteps=10000, callback=callback)

    # setup experiment
    exp = Experiment(flow_params)

    # setup rl policy
    rl_actions = model.next_action

    # run the sumo simulation
    info_dict = exp.run(1, rl_actions=rl_actions)
    print(info_dict)

