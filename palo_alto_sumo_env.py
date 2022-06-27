import numpy as np

from flow.envs.base import Env
from gym.spaces.box import Box


class PaloAltoSumo(Env):
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
        self.agent_route = None
        # remember to change this if you change the get_state() function
        self.state_index_dict = {
            "this_lane": 0,
            "this_speed": 1,
            "max_speed": 2,
            "num_lane": 3,
            "len_target_lane": 4,
            "count_from_right_or_left": 5,
            "dist_to_the_end_of_edge": 6,
            "front_middle_headways": 7,
            "front_middle_speed": 8,
            "front_right_headways": 9,
            "front_right_speed": 10,
            "front_left_headways": 11,
            "front_left_speed": 12,
            "rear_middle_headways": 13,
            "rear_middle_speed": 14,
            "rear_right_headways": 15,
            "rear_right_speed": 16,
            "rear_left_headways": 17,
            "rear_left_speed": 18
        }

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
            rl_agents_ids = self.k.vehicle.get_rl_ids()
            if rl_agents_ids:
                assert rl_agents_ids[0] == "Agent"
                rl_id = rl_agents_ids[0]
                this_lane = self.k.vehicle.get_lane(rl_id)
                lane_follower = self.k.vehicle.get_lane_followers(rl_id)[this_lane]
                if lane_follower:
                    lane_follower_acc = self.k.vehicle.get_realized_accel(lane_follower)
                    if lane_follower_acc < -4:

                        print(lane_follower, ' acc: ', lane_follower_acc)
                else:
                    lane_follower_acc = 0
                crash = self.k.vehicle.get_crash(rl_id) or lane_follower_acc < -4
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

    @property
    def action_space(self):
        """See parent class."""
        return Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    @property
    def observation_space(self):
        """See parent class."""
        return Box(low=-1, high=1, shape=(19,), dtype=np.float32)

    def apply_rl_actions(self, rl_actions=None):
        return self._apply_rl_actions(rl_actions)

    def _apply_rl_actions(self, rl_actions):
        rl_agents_ids = self.k.vehicle.get_rl_ids()
        if rl_agents_ids:
            assert rl_agents_ids[0] == "Agent"
            rl_id = rl_agents_ids[0]
            self.k.vehicle.apply_lane_change(rl_id, self.lc_action_map(rl_actions[1]))
            self.k.vehicle.apply_acceleration(rl_id, rl_actions[0])
        else:
            return

    @staticmethod
    def lc_action_map(lc_action):
        action_map = {(-1000, -0.5): -1, (-0.5, 0.5): 0, (0.5, 1000): 1}
        # -1: change to the right; 1: change to the left; reference: apply_lane_change
        for key in action_map:
            if key[0] < lc_action <= key[1]:
                return action_map[key]

    def compute_reward(self, rl_actions, fail=False, next_s=None, s=None):
        if not next_s.any():
            return 0

        """ Collision Reward """
        r_collision = -10 if fail else 0

        """ Safe: TTC and Time Headway """
        front_dist = next_s[self.state_index_dict["front_middle_headways"]]
        this_speed = next_s[self.state_index_dict["this_speed"]]
        leader_speed = next_s[self.state_index_dict["front_middle_speed"]]
        ttc = front_dist/(this_speed-leader_speed)
        time_headway = front_dist/this_speed if this_speed > 0 else 100
        r_ttc = np.log(ttc/3) if 0 < ttc <= 3 else 0
        r_time_headway = np.exp(-abs(1.75-time_headway)) if 0 < time_headway <= 5 else 0

        """ Efficiency """
        max_speed = next_s[self.state_index_dict["max_speed"]]
        r_efficiency = np.exp(-abs(max_speed-this_speed)) if this_speed < max_speed else -5

        """ Lane Change Reward """
        this_lane = next_s[self.state_index_dict["this_lane"]]
        num_lane = next_s[self.state_index_dict["num_lane"]]
        if rl_actions is None or 0 <= this_lane + rl_actions[1] <= num_lane-1:
            r_lane_change = 0
        else:
            r_lane_change = -10
        # if the following vehicle hard break

        """ Target Lanes Reward (mandatory) """
        len_target_lane = int(next_s[self.state_index_dict["len_target_lane"]])
        count_from_right_or_left = next_s[self.state_index_dict["count_from_right_or_left"]]
        dist_to_the_end_of_edge = next_s[self.state_index_dict["dist_to_the_end_of_edge"]]

        target_lane = [i for i in range(len_target_lane)] if count_from_right_or_left \
            else [num_lane - i - 1 for i in range(len_target_lane)]
        r_target_lane = 5 if this_lane in target_lane else -5

        """ Arrive Bonus """
        if 'Agent' in self.k.vehicle.get_arrived_ids():
            return 100

        w_collision, w_ttc, w_time_headway, w_lc, w_eff = 0.2, 0.2, 0.2, 0.2, 0.2
        w_target_lane = np.exp(-dist_to_the_end_of_edge)
        r = (1 - w_target_lane) * (w_collision * r_collision +
                                   w_ttc * r_ttc +
                                   w_time_headway * r_time_headway +
                                   w_lc * r_lane_change +
                                   w_eff * r_efficiency) + \
            w_target_lane * r_target_lane

        print("r_col {0:6.2f}; r_ttc {1:6.2f}; r_th {2:6.2f}; r_eff {3:6.2f}; r_lc {4:6.2f}; r_tl {5:6.2f}; r {6:6.2f}".
              format(r_collision, r_ttc, r_time_headway, r_efficiency, r_lane_change, r_target_lane, r))

        return r

    def reset(self):
        self.sim_params.restart_instance = True
        super(PaloAltoSumo, self).reset()
        self.env_params.clip_actions = False  # todo some bugs to fix
        self.agent_route = self.k.network.rts["Agent"][0][0]

    def get_state(self, **kwargs):
        """See class definition."""
        rl_agents_ids = self.k.vehicle.get_rl_ids()
        if rl_agents_ids:
            assert rl_agents_ids[0] == "Agent"
            """ State of agent """
            # [lane_number, v_x, lane_max_speed]
            rl_id = rl_agents_ids[0]
            this_edge = self.k.vehicle.get_edge(rl_id)
            this_lane = self.k.vehicle.get_lane(rl_id)

            this_speed = self.k.vehicle.get_speed(rl_id)
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
                            if next_edge in self.agent_route and lane not in target_lane:
                                target_lane.append(lane)
                    except IndexError:
                        continue
            if 0 in target_lane:  # right
                count_from_right_or_left = 1  # count from right
            else:
                count_from_right_or_left = 0  # count from left
            dist_to_the_end_of_edge = self.k.network.edge_length(this_edge) - self.k.vehicle.get_position(rl_id)
            assert len(target_lane) <= num_lanes
            stay_merge_or_exit = [num_lanes, len(target_lane), count_from_right_or_left, dist_to_the_end_of_edge]

            """ Affordance cars """
            # Front cars: [this, right, left]
            leaders = self.k.vehicle.get_lane_leaders(rl_id)  # in all lanes
            leaders_speed = self.k.vehicle.get_lane_leaders_speed(rl_id)
            headways = self.k.vehicle.get_lane_headways(rl_id)
            front_cars_state = [headways[this_lane], leaders_speed[this_lane]]
            # Front right
            if this_lane-1 > 0 and leaders[this_lane-1]:
                front_cars_state.extend([headways[this_lane-1], leaders_speed[this_lane-1]])
            else:
                front_cars_state.extend([1000, -1001])
            # Front left
            if this_lane+1 < num_lanes and leaders[this_lane+1]:
                front_cars_state.extend([headways[this_lane+1], leaders_speed[this_lane+1]])
            else:
                front_cars_state.extend([1000, -1001])

            # Rear cars: Front cars: [this, right, left]
            followers = self.k.vehicle.get_lane_followers(rl_id)  # in all lanes
            followers_speed = self.k.vehicle.get_lane_followers_speed(rl_id)
            tailways = self.k.vehicle.get_lane_tailways(rl_id)
            rear_cars_state = [tailways[this_lane], followers_speed[this_lane]]
            # Rear right
            if this_lane-1 > 0 and followers[this_lane-1]:
                rear_cars_state.extend([tailways[this_lane-1], followers_speed[this_lane-1]])
            else:
                rear_cars_state.extend([1000, -1001])
            # Rear left
            if this_lane+1 < num_lanes and followers[this_lane+1]:
                rear_cars_state.extend([tailways[this_lane+1], followers_speed[this_lane+1]])
            else:
                rear_cars_state.extend([1000, -1001])

            """ State """
            state = this_car_state + stay_merge_or_exit + front_cars_state + rear_cars_state
            state = np.array(state, dtype=np.float32)
            return state
        else:
            print("Agent not joined yet")
            return None
