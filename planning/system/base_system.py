import os
import pickle
import numpy as np
import cv2

from stap.envs.pybullet.table.primitives import Pick, Pull, Push, Place
from planning.graphs.graph_manager import GraphManager
from stap.envs.pybullet.table_env import TableEnv

from utils.general_utils import save_pickle
from planning.utils.env_utils import primitive_name_from_action, load_env, fix_high_level_action

from planning.high_level.catalog import HIGH_LEVEL_PLANNERS
from planning.low_level.catalog import Low_LEVEL_PLANNERS

PRIMITIVE_CATALOG = {
    'pick': Pick,
    'place': Place,
    'pull': Pull,
    'push': Push,
}


class BaseSystemPlanner():
    def __init__(self, cfg:dict):
        
        self.cfg = cfg
        self.graph_manager = GraphManager()
        self._load_low_level_planner()
        self._load_high_level_planner()
        self._load_env(
            task_path=self.cfg.GENERAL_PARMAS.task_path,
            seed=self.cfg.GENERAL_PARMAS.seed
        )
        ##initial state image
        img = self.env.render()
        cv2.imwrite(os.path.join(self.cfg['GENERAL_PARMAS'].output_path, "image.png"), img)
         
        self._init_primitives()
        self._init_logger()
        
        #complete list
        self.complete_list = []
        self.black_list = []
        
        self.step = 0
        self.max_steps = self.cfg.GENERAL_PARMAS.max_steps
        self.max_high_level_steps = self.cfg.GENERAL_PARMAS.max_high_level_steps
    

    def _execute_low_level_action(self):
        raise NotImplementedError
        
    def run(self):
        raise NotImplementedError

    def _init_logger(self):
        self.logger = {}
        self.logger_cnt = 0
        
    def _save_graph(self, path):
        with open(os.path.join(path, "high_level_grpah.gpickle"), 'wb') as f:
            pickle.dump(self.graph_manager.high_level_graph, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(path, "low_level_grpah.gpickle"), 'wb') as f:
            pickle.dump(self.graph_manager.low_level_graph, f, pickle.HIGHEST_PROTOCOL)
        
    def _log(self, key:str, value):
        if key not in self.logger:
            self.logger[key] = []
        if isinstance(value, dict):
            log = {
                "index": self.logger_cnt
            }
            for item in value:
                log[item] = value[item]
        else:
            raise

        self.logger[key].append(log)
           
    def _load_low_level_planner(self):
        self.low_level_planner = Low_LEVEL_PLANNERS[self.cfg.LOW_LEVEL_PLANNER.type](
            params=self.cfg.LOW_LEVEL_PLANNER.params
            )
        
    def _load_high_level_planner(self):
        self.high_level_planner = HIGH_LEVEL_PLANNERS[self.cfg.HIGH_LEVEL_PLANNER.type](
            params=self.cfg.HIGH_LEVEL_PLANNER.params
            )
        _ = self.high_level_planner.init_search(
            pddl_domain=self.cfg.GENERAL_PARMAS.pddl_domain, 
            pddl_problem=self.cfg.GENERAL_PARMAS.pddl_problem,
            search_method=self.cfg.GENERAL_PARMAS.search_method,
            heuristic_method=self.cfg.GENERAL_PARMAS.heuristic_name
        )
        output = self.high_level_planner.run_one_step()
        self._update_graph_from_high_level_planner_output(output=output)
        
    def _init_primitives(self):
        self.primitives = {}
        for primitive in self.env._primitives:
            self.primitives[primitive] = PRIMITIVE_CATALOG[primitive]
     
    def _load_env(self, task_path:str, seed:int):
        env, initial_observation, initial_info = load_env(
            task_path=task_path,
            seed=seed
        )
        self.env = env
        self.initial_observation = initial_observation
        self.initial_info = initial_info
        robot_state = self.env.get_state()

        self.graph_manager.add_low_level_node(
            initial_high_level_state=None,
            high_level_state=self.high_level_planner.root.state,
            high_level_action=None,
            low_level_action=None,
            initial_low_level_state=None,
            low_level_state=self.initial_observation,
            goal_reached=False,
            info=self.initial_info,
            robot_state=robot_state,
        )

    def _get_primitive(self, action:str, env:TableEnv):
        action_name, objects = primitive_name_from_action(action=action)
        arg_objects = [env.objects[obj_name] for obj_name in objects]
        idx_policy = env.primitives.index(action_name)
        primitive = self.primitives[action_name](
            env=env,
            idx_policy=idx_policy,
            arg_objects=arg_objects,
        )
        
        return primitive

    def _update_graph_from_high_level_planner_output(self, output:dict):
        if output['node'].action is not None:
            output['node'].action.name = fix_high_level_action(
                high_level_action=output['node'].action.name
            )
        id = self.graph_manager.add_high_level_node(
            high_level_state=output['node'],
            goal_reached=output["found_goal"]
        )
        return id

    def _increase_node_count(self, node_id:int, action_id:int):
        if action_id not in self.graph_manager.low_level_graph.graph.nodes._nodes[node_id]["tries"]:
            self.graph_manager.low_level_graph.graph.nodes._nodes[node_id]["tries"][action_id] = 0
        self.graph_manager.low_level_graph.graph.nodes._nodes[node_id]["tries"][action_id] +=1

    def _select_low_level_state(self, high_level_state):
        low_level_states =\
            self.graph_manager.get_all_low_level_states_from_high_level_state(state=high_level_state)
        
        """
        if self.cfg.GENERAL_PARMAS.filter_isComplete:
            filter_list = []
            for state in low_level_states:
                if not state['IsComplete']:
                    filter_list.append(state)
            if len(filter_list) > 0:
                low_level_states = filter_list
        """
        if len(low_level_states) == 0:
            return None, None, None, None
        probs = np.array([(1/(sum(state['tries'].values())+1)) for state in low_level_states])
        probs /= sum(probs)
                
        low_level_state = np.random.choice(low_level_states, p=probs)
        
        return low_level_state['low_level_state'], low_level_state['info'], low_level_state['robot_state'], low_level_state['index']

    def _prune_high_level_node(self, high_level_state_id, high_level_action_id, low_level_state_id):
        if self.graph_manager.low_level_graph.graph.nodes._nodes[low_level_state_id]["tries"][high_level_action_id] >=\
            self.cfg.GENERAL_PARMAS.max_tries_per_node:
            ids_to_delete = []
            state = self.graph_manager.high_level_graph.graph.nodes._nodes[high_level_action_id]["state"]
            ids_to_delete.append(high_level_action_id)
            if state not in self.black_list:
                self.black_list.append(state)
            for node_key in self.graph_manager.high_level_graph.graph.nodes._nodes:
                node = self.graph_manager.high_level_graph.graph.nodes._nodes[node_key]
                parent = node['parent']
                if parent in ids_to_delete:
                    state = self.graph_manager.high_level_graph.graph.nodes._nodes[node_key]["state"]
                    if state not in self.black_list:
                        self.black_list.append(state)
                    ids_to_delete.append(node_key)
            queue_nodes_to_delete = []
            for queue_node in self.high_level_planner.queue:
                if queue_node.parent.state in self.black_list:
                    queue_nodes_to_delete.append(queue_node)
            for node in queue_nodes_to_delete:
                self.high_level_planner.queue.remove(node)

    def _execute_high_level_action(self):
        results = self.high_level_planner.run_one_step()
        if results['node'] is not None and results["queue_isnt_empty"]:
            id = self._update_graph_from_high_level_planner_output(output=results)
            results["id"] = id
            
        results["type"] = "high_level",
        
            
        return results
    
    def _action_validator(self, action):
        words = action.split(" ")
        if words[1] in words[2]:
            return False
        return True

    def save(self, time:float=None):
        if time is not None:
            self.logger["total_time"] = time
        self.logger["allowed_time"] = self.cfg.GENERAL_PARMAS.max_time 
        save_pickle(os.path.join(self.cfg['GENERAL_PARMAS'].output_path, "log.pickle"), self.logger)
        self._save_graph(path=self.cfg['GENERAL_PARMAS'].output_path)