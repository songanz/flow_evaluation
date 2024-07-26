import numpy as np
import random
from copy import deepcopy

from gym.spaces.box import Box
from .palo_alto_sumo_env import PaloAltoSumo


class PaloAltoSumoAtt(PaloAltoSumo):
    """Palo Alto highway environment used to run simulations in the absence of autonomy.

    Required from env_params
        None

    Optional from env_params
        reward_fn : A reward function which takes an an input the environment
        class and returns a real number.

    States
        States are an empty list.

    Actions
        No actions are provided to any RL agent.

    Rewards
        The reward is zero at every step.

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    def __init__(self,
                 env_params,
                 sim_params,
                 network=None,
                 simulator='traci',
                 scenario=None):
        super().__init__(env_params, sim_params, network, simulator, scenario)
        self.ego_vehicle = None
        self.ego_veh_model_path = None
        self.ego_veh_model = None
        self.ego_vehicle_id = "Agent"
        self.agent_id = "Attacker"

    @property
    def observation_space(self):
        """See parent class."""
        return Box(low=-float("inf"), high=float("inf"), shape=(24,), dtype=np.float32)

    def load_ego_vehicle(self, ego_veh_model_path, ego_veh_model):
        self.ego_veh_model_path = ego_veh_model_path
        self.ego_veh_model = ego_veh_model
        self.ego_vehicle = self.ego_veh_model.load(self.ego_veh_model_path)

    def apply_rl_actions(self, rl_actions=None):
        return self._apply_rl_actions(rl_actions)

    def _apply_rl_actions(self, rl_actions):
        if rl_actions is None:
            return None
        rl_agents_ids = self.k.vehicle.get_rl_ids()
        if rl_agents_ids:
            for id_ in rl_agents_ids:
                if id_ == "Attacker":
                    self.k.vehicle.apply_acceleration(id_, rl_actions[0])
                    self.k.vehicle.apply_lane_change(id_, self.lc_action_map(rl_actions[1]))
                elif id_ == "Agent":
                    ego_state = self.get_ego_state()
                    actions, _ = self.ego_vehicle.predict(ego_state)
                    self.k.vehicle.apply_acceleration(id_, actions[0])
                    self.k.vehicle.apply_lane_change(id_, self.lc_action_map(actions[1]))
                else:
                    print("Unknow RL agent: ", id_)
        else:
            return

    def step(self, rl_actions):
        crash = False
        obs = np.copy(self.get_state())
        for _ in range(self.env_params.sims_per_step):
            self.time_counter += 1
            self.step_counter += 1

            # perform acceleration actions for controlled human-driven vehicles
            if len(self.k.vehicle.get_controlled_ids()) > 0:
                accel = []
                for veh_id in self.k.vehicle.get_controlled_ids():
                    action = self.k.vehicle.get_acc_controller(
                        veh_id).get_action(self)
                    accel.append(action)
                self.k.vehicle.apply_acceleration(
                    self.k.vehicle.get_controlled_ids(), accel)

            # perform lane change actions for controlled human-driven vehicles
            if len(self.k.vehicle.get_controlled_lc_ids()) > 0:
                direction = []
                for veh_id in self.k.vehicle.get_controlled_lc_ids():
                    target_lane = self.k.vehicle.get_lane_changing_controller(
                        veh_id).get_action(self)
                    direction.append(target_lane)
                self.k.vehicle.apply_lane_change(
                    self.k.vehicle.get_controlled_lc_ids(),
                    direction=direction)

            # perform (optionally) routing actions for all vehicles in the
            # network, including RL and SUMO-controlled vehicles
            routing_ids = []
            routing_actions = []
            for veh_id in self.k.vehicle.get_ids():
                if self.k.vehicle.get_routing_controller(veh_id) \
                        is not None:
                    routing_ids.append(veh_id)
                    route_contr = self.k.vehicle.get_routing_controller(
                        veh_id)
                    routing_actions.append(route_contr.choose_route(self))

            self.k.vehicle.choose_routes(routing_ids, routing_actions)

            self.apply_rl_actions(rl_actions)

            self.additional_command()

            # advance the simulation in the simulator by one step
            self.k.simulation.simulation_step()

            # store new observations in the vehicles and traffic lights class
            self.k.update(reset=False)

            # update the colors of vehicles
            if self.sim_params.render:
                self.k.vehicle.update_vehicle_colors()

            # crash: wiil not terminate if other vehicel crash
            # todo: get crash of the agent and crash of the attacker
            rl_agents_ids = self.k.vehicle.get_rl_ids()
            if self.ego_vehicle_id in rl_agents_ids and self.agent_id in rl_agents_ids:
                if self.k.vehicle.get_crash(self.ego_vehicle_id):
                    crash = True
                    print('Agent crashed')
                    break
            else:
                crash = False

            # stop collecting new simulation steps if there is a collision
            if crash:
                break

            # render a frame
            self.render()

        states = self.get_state()

        # collect information of the state of the network based on the
        # environment class used
        self.state = np.asarray(states).T

        # collect observation new state associated with action
        next_observation = np.copy(states)

        # test if the environment should terminate due to a collision or the
        # time horizon being met
        done = (self.time_counter >= self.env_params.sims_per_step *
                (self.env_params.warmup_steps + self.env_params.horizon)
                or crash)

        # compute the info for each agent
        infos = {}

        # compute the reward
        if self.env_params.clip_actions:
            rl_clipped = self.clip_actions(rl_actions)
            reward = self.compute_reward(rl_clipped, fail=crash, next_s=next_observation, s=obs)
        else:
            reward = self.compute_reward(rl_actions, fail=crash, next_s=next_observation, s=obs)

        return next_observation, reward, done, infos

    def compute_reward(self, rl_actions, fail=False, next_s=None, s=None):
        if not next_s.any():
            return 0

        """ Collision Reward """
        if fail:
            print("Failed, r_collision: -10")
            return -10

        """ Safe: TTC and Time Headway """
        front_dist = next_s[self.state_index_dict["front_middle_headways"]]
        this_speed = next_s[self.state_index_dict["this_speed"]]
        leader_speed = next_s[self.state_index_dict["front_middle_speed"]]
        ttc = front_dist / (this_speed - leader_speed)
        time_headway = front_dist / this_speed if this_speed > 0 else 100
        r_ttc = np.log(ttc / 3) if 0 < ttc <= 3 else 0
        r_time_headway = np.exp(-abs((1.75 - time_headway)**2)/2) if time_headway > 0 else 0

        """ Efficiency """
        max_speed = next_s[self.state_index_dict["max_speed"]]
        r_efficiency = np.exp(-abs((max_speed - this_speed)**2)/30) if this_speed <= max_speed else -1

        """ Lane Change Reward """
        this_lane = next_s[self.state_index_dict["this_lane"]]
        num_lane = next_s[self.state_index_dict["num_lane"]]
        if rl_actions is None or 0 <= this_lane + rl_actions[1] <= num_lane - 1:
            r_lane_change = 1
        else:
            r_lane_change = -1
        # if the following vehicle hard break

        """ Target Lanes Reward (mandatory) """
        len_target_lane = int(next_s[self.state_index_dict["len_target_lane"]])
        count_from_right_or_left = next_s[self.state_index_dict["count_from_right_or_left"]]
        dist_to_the_end_of_edge = next_s[self.state_index_dict["dist_to_the_end_of_edge"]]

        target_lane = [i for i in range(len_target_lane)] if count_from_right_or_left \
            else [num_lane - i - 1 for i in range(len_target_lane)]
        r_target_lane = 2 if this_lane in target_lane else -2

        """ Arrive Bonus """
        if 'Agent' in self.k.vehicle.get_arrived_ids():
            print("Agent arrives, r_arrive: 100")
            return 100

        w_ttc, w_time_headway, w_lc, w_eff = 0.25, 0.25, 0.25, 0.25
        w_target_lane = np.exp(-dist_to_the_end_of_edge)
        r = (1 - w_target_lane) * (w_ttc * r_ttc +
                                   w_time_headway * r_time_headway +
                                   w_lc * r_lane_change +
                                   w_eff * r_efficiency) + \
            w_target_lane * r_target_lane

        print("r_ttc {0:6.2f}; r_th {1:6.2f}; r_eff {2:6.2f}; r_lc {3:6.2f}; r_tl {4:6.2f}; r {5:6.2f}".
              format(r_ttc, r_time_headway, r_efficiency, r_lane_change, r_target_lane, r))

        return r

    def reset(self):
        # reset the time counter
        self.time_counter = 0

        # Now that we've passed the possibly fake init steps some rl libraries
        # do, we can feel free to actually render things
        if self.should_render:
            self.sim_params.render = True
            # got to restart the simulation to make it actually display anything
            self.restart_simulation(self.sim_params)

        if self.sim_params.restart_instance or \
                (self.step_counter > 2e6 and self.simulator != 'aimsun'):
            self.step_counter = 0
            # issue a random seed to induce randomness into the next rollout
            self.sim_params.seed = random.randint(0, int(1e5))

            self.k.vehicle = deepcopy(self.initial_vehicles)
            self.k.vehicle.master_kernel = self.k
            # restart the sumo instance
            self.restart_simulation(self.sim_params)

        # perform shuffling (if requested)
        elif self.initial_config.shuffle:
            self.setup_initial_state()

        # clear all vehicles from the network and the vehicles class
        if self.simulator == 'traci':
            for veh_id in self.k.kernel_api.vehicle.getIDList():  # FIXME: hack
                self.k.vehicle.remove(veh_id)

        # clear all vehicles from the network and the vehicles class
        # FIXME (ev, ak) this is weird and shouldn't be necessary
        for veh_id in list(self.k.vehicle.get_ids()):
            # do not try to remove the vehicles from the network in the first
            # step after initializing the network, as there will be no vehicles
            if self.step_counter == 0:
                continue
            self.k.vehicle.remove(veh_id)

        # do any additional resetting of the vehicle class needed
        self.k.vehicle.reset()

        # reintroduce the initial vehicles to the network
        for veh_id in self.initial_ids:
            type_id, edge, lane_index, pos, speed = \
                self.initial_state[veh_id]

            self.k.vehicle.add(
                veh_id=veh_id,
                type_id=type_id,
                edge=edge,
                lane=lane_index,
                pos=pos,
                speed=speed)

        # advance the simulation in the simulator by one step
        self.k.simulation.simulation_step()

        # update the information in each kernel to match the current state
        self.k.update(reset=True)

        # update the colors of vehicles
        if self.sim_params.render:
            self.k.vehicle.update_vehicle_colors()

        states = self.get_state()

        # collect information of the state of the network based on the
        # environment class used
        self.state = np.asarray(states).T

        # observation associated with the reset (no warm-up steps)
        observation = np.copy(states)

        # perform (optional) warm-up steps before training
        for _ in range(self.env_params.warmup_steps):
            observation, _, _, _ = self.step(rl_actions=None)
        while not self.k.vehicle.get_rl_ids() \
                and ("Agent" not in self.k.vehicle.get_rl_ids()
                     or "Attacker" not in self.k.vehicle.get_rl_ids()):
            observation, _, _, _ = self.step(rl_actions=None)
        print("Observation 0 after warm-up", observation)

        # render a frame
        self.render(reset=True)

        return observation

    def get_state(self, **kwargs):
        rl_agents_ids = self.k.vehicle.get_rl_ids()
        if rl_agents_ids and "Attacker" in rl_agents_ids:
            attacker_state = self._get_state_by_id("Attacker")  # list
            """ ego vehicle position, speed and action"""
            ego_speed = self.k.vehicle.get_speed("Agent")
            rel_x = self.k.vehicle.get_2d_position("Attacker")[0] - self.k.vehicle.get_2d_position("Agent")[0]
            rel_y = self.k.vehicle.get_2d_position("Attacker")[1] - self.k.vehicle.get_2d_position("Agent")[1]
            actions, _ = self.ego_vehicle.predict(self.get_ego_state())
            state = [ego_speed, rel_x, rel_y] + list(actions) + attacker_state
            return state
        else:
            return None

    def get_ego_state(self, **kwargs):
        """See class definition."""
        rl_agents_ids = self.k.vehicle.get_rl_ids()
        if rl_agents_ids and "Agent" in rl_agents_ids:
            return np.array(self._get_state_by_id("Agent"), dtype=np.float32)
        else:
            print("Ego vehicle has not joined yet")
            return None

    def _get_state_by_id(self, v_id):
        """ Get state by id """
        v_route = self.k.network.rts[v_id][0][0]
        """ State of this vehicle """
        # [lane_number, v_x, lane_max_speed]
        this_edge = self.k.vehicle.get_edge(v_id)
        this_lane = self.k.vehicle.get_lane(v_id)

        this_speed = self.k.vehicle.get_speed(v_id)
        max_speed = self.k.network.speed_limit(this_edge)
        this_car_state = [this_lane, this_speed, max_speed]

        """ Target lane & target dist """
        # [num_lanes, len(target_lane), count_from_left_or_right, dist_to_the_end_of_edge]
        num_lanes = self.k.network.num_lanes(this_edge)  # get num of lanes at current edge
        # Fixed: something wrong with loading the net xml, sometime it gets larger num_lanes!
        target_lane = []
        for lane in range(num_lanes):
            junctions = self.k.network.next_edge(this_edge, lane)
            if not junctions:
                # print("No next junction to edge: ", this_edge, " at lane: ", lane)
                pass
            for junction in junctions:
                try:
                    for next_edge, _ in self.k.network.next_edge(junction[0], junction[1]):
                        if next_edge in v_route and lane not in target_lane:
                            target_lane.append(lane)
                except IndexError:
                    continue
        if 0 in target_lane:  # right
            count_from_right_or_left = 1  # count from right
        else:
            count_from_right_or_left = 0  # count from left
        dist_to_the_end_of_edge = self.k.network.edge_length(this_edge) - self.k.vehicle.get_position(v_id)
        assert len(target_lane) <= num_lanes
        stay_merge_or_exit = [num_lanes, len(target_lane), count_from_right_or_left, dist_to_the_end_of_edge]

        """ Affordance cars """
        # Front cars: [this, right, left]
        leaders = self.k.vehicle.get_lane_leaders(v_id)  # in all lanes
        leaders_speed = self.k.vehicle.get_lane_leaders_speed(v_id)
        headways = self.k.vehicle.get_lane_headways(v_id)
        front_cars_state = [headways[this_lane], leaders_speed[this_lane]]
        # Front right
        if this_lane - 1 > 0 and leaders[this_lane - 1]:
            front_cars_state.extend([headways[this_lane - 1], leaders_speed[this_lane - 1]])
        else:
            front_cars_state.extend([1000, -1001])
        # Front left
        if this_lane + 1 < num_lanes and leaders[this_lane + 1]:
            front_cars_state.extend([headways[this_lane + 1], leaders_speed[this_lane + 1]])
        else:
            front_cars_state.extend([1000, -1001])

        # Rear cars: Front cars: [this, right, left]
        followers = self.k.vehicle.get_lane_followers(v_id)  # in all lanes
        followers_speed = self.k.vehicle.get_lane_followers_speed(v_id)
        tailways = self.k.vehicle.get_lane_tailways(v_id)
        rear_cars_state = [tailways[this_lane], followers_speed[this_lane]]
        # Rear right
        if this_lane - 1 > 0 and followers[this_lane - 1]:
            rear_cars_state.extend([tailways[this_lane - 1], followers_speed[this_lane - 1]])
        else:
            rear_cars_state.extend([1000, -1001])
        # Rear left
        if this_lane + 1 < num_lanes and followers[this_lane + 1]:
            rear_cars_state.extend([tailways[this_lane + 1], followers_speed[this_lane + 1]])
        else:
            rear_cars_state.extend([1000, -1001])

        """ State """
        state = this_car_state + stay_merge_or_exit + front_cars_state + rear_cars_state
        return state
