import random
import datetime
import numpy as np

from planning.system.base_system import BaseSystemPlanner

from planning.utils.env_utils import fix_high_level_action

class SystemPlanner(BaseSystemPlanner):
    def __init__(self, cfg:dict):
        super().__init__(cfg=cfg)

    def _select_plan(self, all_plans:list):
        return all_plans[0]

    def _select_high_level_action(self, high_level_state_id):
        edges = self.graph_manager.high_level_graph.graph.edges._adjdict
        selected_high_level_action_edge_id =\
            random.choice(list(edges[high_level_state_id].keys()))
        
        selected_high_level_action_edge = edges[high_level_state_id][selected_high_level_action_edge_id]
        goal_reached = self.graph_manager.high_level_graph.graph.nodes._nodes[selected_high_level_action_edge_id]['goal_reached']
        
        return selected_high_level_action_edge['high_level_action'], selected_high_level_action_edge_id, goal_reached

    def _select_high_level_state(self, has_low_level:bool=True, has_high_level_action:bool=False):
        """
        select high level state to start from

        Args:
            has_low_level (bool, optional): _description_. Defaults to True.
            has_high_level_action (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        if has_low_level:
            options = []
            for node in self.graph_manager.low_level_graph.graph.nodes._nodes.values():
                if node['high_level_node_id'] not in options:
                    #if we need high level action and not high level action, continue
                    if has_high_level_action and\
                        len(self.graph_manager.high_level_graph.graph.edges._adjdict[node['high_level_node_id']]) == 0:
                        continue
                    options.append(node['high_level_node_id'])
        else:
            options = list(self.graph_manager.high_level_graph.graph.nodes._nodes.keys())
        if len(options) == 0:
            return None, None
        
        #filter black list
        old_options_len = len(options)
        new_options = []
        for option in options:
            state = self.graph_manager.high_level_graph.graph.nodes._nodes[option]["state"]
            if state not in self.black_list:
                new_options.append(option)
        options = new_options
        
        print("removed_options = ", old_options_len - len(new_options))
        
        if len(options) < 1:
            temp=1
        
        selected_high_level_state_id = random.choice(options)
        selected_high_level_state = \
            self.graph_manager.high_level_graph.graph.nodes._nodes[selected_high_level_state_id]["state"]
        
        return selected_high_level_state, selected_high_level_state_id

    def _find_id(self, high_level_state_id, high_level_action_id, ids_to_delete):
        action = self.graph_manager.high_level_graph.graph.edges._adjdict[high_level_state_id][high_level_action_id]['high_level_action']
        for node_key in self.graph_manager.high_level_graph.graph.nodes._nodes:
            node = self.graph_manager.high_level_graph.graph.nodes._nodes[node_key]
            if node['parent'] is not None and node['parent'] == high_level_state_id:
                done, ids_to_delete = self._find_id(self, node_key, high_level_action_id, ids_to_delete)

    def _execute_low_level_action(self):
        didnt_found_yet = True
        cnt = 0
        while didnt_found_yet:
            if cnt >= self.cfg.GENERAL_PARMAS.max_tries_in_loop:
                return None
            cnt += 1
            #select high level state
            high_level_state, high_level_state_id =\
                self._select_high_level_state(has_low_level=True, has_high_level_action=True)
            if high_level_state is None:
                return None
            
            #select high level action
            high_level_action, high_level_action_id, goal_reached =\
                self._select_high_level_action(high_level_state_id=high_level_state_id)
            high_level_action = fix_high_level_action(
                high_level_action=high_level_action
            )
            
            #select low level state
            observation, info, robot_state, low_level_state_id =\
                self._select_low_level_state(high_level_state=high_level_state)
            
            #check if this triple is already done
            if self.cfg.GENERAL_PARMAS.filter_isComplete:
                sample = {
                    "high_level_state": high_level_state,
                    "high_level_action": high_level_action,
                    "low_level_state": observation,
                }
                if sample in self.complete_list:
                    continue
                else:
                    didnt_found_yet = False
            else:
                didnt_found_yet = False
        
        self._increase_node_count(
            node_id=low_level_state_id,
            action_id=high_level_action_id,
            )
        
        action_is_ok = self._action_validator(high_level_action)
        if not action_is_ok:
            return {
                "type": "low_level",
                "goal_reached": False,
            }
        
        self.env.set_observation(observation)
        self.env.set_state(state=robot_state)
        primitive = self._get_primitive(
            action=high_level_action,
            env=self.env
        )
        
        #get action from low level plannes
        self.env.set_primitive(primitive=primitive)
        low_level_action = self.low_level_planner.get_action(
            obs0=observation,
            observation_indices= info['policy_args']['observation_indices'],
            raw_high_level_action=high_level_action,
            device=self.cfg.GENERAL_PARMAS.device,
            primitive=primitive
        )

        #execute low level action
        try:
            end_observation, reward, terminated, truncated, info = self.env.step(low_level_action)
        except Exception as e:
            raise
        
        if reward > 0.0:
            self.graph_manager.add_low_level_node(
                initial_high_level_state=high_level_state,
                high_level_state =\
                    self.graph_manager.high_level_graph.graph.nodes._nodes[high_level_action_id]["state"],
                high_level_action=high_level_action,
                low_level_action=low_level_action,
                initial_low_level_state=observation,
                low_level_state=end_observation,
                goal_reached=goal_reached,
                info=info,
                robot_state=self.env.get_state()
            )
            if self.cfg.GENERAL_PARMAS.filter_isComplete:
                self.complete_list.append(sample)
            
        else:
            self._prune_high_level_node(
                high_level_state_id = high_level_state_id,
                high_level_action_id = high_level_action_id,
                low_level_state_id = low_level_state_id,
            )
        
        results = {
            "type": "low_level",
            "end_observation": end_observation,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
            "goal_reached": goal_reached and reward > 0.0
        }
        
        return results

    def _update_queue(self):
        queue = self.high_level_planner.queue
        
        #right now I am not doing nothing, but we will want to change it
        new_queue = queue
        
        self.high_level_planner.queue = new_queue

    def run(self):
        options = ["high", "low"]
        #selected_step = random.choice(options)
        selected_step = np.random.choice(
            options,
            p=[self.cfg.GENERAL_PARMAS.p_high_level, 1-self.cfg.GENERAL_PARMAS.p_high_level]
        )
        start_time = datetime.datetime.now()
        if selected_step == "low":
            results = self._execute_low_level_action()
        elif selected_step == "high":
            results = self._execute_high_level_action()
        else:
            raise
        if results is None:
            return None
        self._update_queue()
        
        #logging
        results["selected_step"] = selected_step
        time = datetime.datetime.now()-start_time
        results["time"] = time.seconds + time.microseconds/1000000
        self._log(key="results", value=results)
        self.logger_cnt += 1

        return results