import datetime
import numpy as np
from collections import deque 


from planning.system.base_system import BaseSystemPlanner

class ELSSystemPlanner(BaseSystemPlanner):
    def __init__(self, cfg:dict):
        super().__init__(cfg=cfg)
        
        self.queue = []
        self.step = 0
        self.max_steps = 1000
        for _ in range(len(self.high_level_planner.queue)):
            start_time = datetime.datetime.now()
            results = self._execute_high_level_action()
            self.step +=1 
            #logging
            results["selected_step"] = "high"
            time = datetime.datetime.now()-start_time
            results["time"] = time.seconds + time.microseconds/1000000
            self._log(key="results", value=results)
            self.logger_cnt += 1
            sample = self._get_empty_sample()
            assert len(self.graph_manager.low_level_graph.graph.nodes._nodes) == 1
            sample["low_level_state"] = self.graph_manager.low_level_graph.graph.nodes._nodes[0]['low_level_state']
            sample["high_level_action"] = results['node'].action.name
            sample["high_level_next_state"] = results['id']
            self.queue.append(sample)
        
        self.solutions = []
        self.time_limit = 1000000
        self.time = 0
        self.pc = self.w0 = self.pw = 1
        self.epslion = 0
        self.co = 0.0001
     
    def _get_empty_sample(self):
        sample = {
            "low_level_state": None,
            "high_level_action": None,
            "high_level_next_state": None,
        }
        return sample
     
    def _pop(self):
        nodes_f = []
        for node in self.queue:
            id = self.graph_manager.low_level_graph.get_id_from_state(state=node['low_level_state'])
            nodes_f.append(self._compute_f(node=id))
        index = np.argmin(np.array(nodes_f))
        node_with_lowest_fn = self.queue.pop(index)
        return node_with_lowest_fn
    
    def _widen(self, node):
        id = self.graph_manager.low_level_graph.get_id_from_state(state=node['low_level_state'])
        cn = sum(self.graph_manager.low_level_graph.graph.nodes._nodes[id]['tries'])
        if cn == 0 and node.parent.branching_factor == np.inf:
            sibling = self.get_sibling(node=node)
        else:
            sibling = None
        
        return sibling
        
    def _get_high_level_action_id(self, action, next_state_id):
        options = self.graph_manager.high_level_graph.graph.out_degree._pred[next_state_id]
        for state_id in options:
            if options[state_id]["high_level_action"] == action:
                return state_id

    def _execute_low_level_action(self, low_level_action:np.ndarray):
        
        #execute low level action
        try:
            end_observation, reward, terminated, truncated, info = self.env.step(low_level_action)
        except Exception as e:
            raise

        return end_observation, reward, terminated, truncated, info

    def _compute(self, node):
        self.step +=1
        id = self.graph_manager.low_level_graph.get_id_from_state(state=node['low_level_state'])
        high_level_node_id =self.graph_manager.low_level_graph.graph.nodes._nodes[id]['high_level_node_id']
        high_level_action_id = self._get_high_level_action_id(
            action=node['high_level_action'],
            next_state_id=node['high_level_next_state']
        )
        
        if high_level_node_id is None:
            high_level_state = None
        else:
            high_level_state = self.graph_manager.high_level_graph.graph.nodes._nodes[high_level_node_id]["state"]
        self._increase_node_count(
            node_id=id,
            action_id=high_level_action_id,
            )
        
        action_is_ok = self._action_validator(node['high_level_action'])
        if not action_is_ok:
            return {
                "type": "low_level",
                "goal_reached": False,
            }
        
        self.env.set_observation(node['low_level_state'])
        self.env.set_state(state=self.graph_manager.low_level_graph.graph.nodes._nodes[id]['robot_state'])
        primitive = self._get_primitive(
            action=node['high_level_action'],
            env=self.env
        )
        
        info = self.graph_manager.low_level_graph.graph.nodes._nodes[id]['info']
        
        #get action from low level plannes
        self.env.set_primitive(primitive=primitive)
        low_level_action = self.low_level_planner.get_action(
            obs0=node['low_level_state'],
            observation_indices= info['policy_args']['observation_indices'],
            raw_high_level_action=node['high_level_action'],
            device=self.cfg.GENERAL_PARMAS.device,
            primitive=primitive
        )

        #execute low level action
        end_observation, reward, terminated, truncated, info =\
            self._execute_low_level_action(low_level_action=low_level_action)
        
        goal_reached = self.graph_manager.high_level_graph.graph.nodes._nodes[node['high_level_next_state']]['goal_reached']
        if reward > 0.0:
            self.graph_manager.add_low_level_node(
                initial_high_level_state=high_level_state,
                high_level_state =\
                    self.graph_manager.high_level_graph.graph.nodes._nodes[node['high_level_next_state']]["state"],
                high_level_action=node['high_level_action'],
                low_level_action=low_level_action,
                initial_low_level_state=node['low_level_state'],
                low_level_state=end_observation,
                goal_reached=goal_reached,
                info=info,
                robot_state=self.env.get_state()
            )
            
        else:
            self._prune_high_level_node(
                high_level_state_id = high_level_node_id,
                high_level_action_id = high_level_action_id,
                low_level_state_id = id,
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

    def _deepen(self, results):
        output = []
        id = self.graph_manager.low_level_graph.get_id_from_state(state=results['end_observation'])
        high_level_node_id =self.graph_manager.low_level_graph.graph.nodes._nodes[id]['high_level_node_id']
        high_level_state = self.graph_manager.high_level_graph.graph.nodes._nodes[high_level_node_id]['state']
        get_all_action = []
        new_queue = []
        cnt = 0
        for queue_node in self.high_level_planner.queue:
            if queue_node.parent.state == high_level_state:
                get_all_action.insert(0, queue_node)
                cnt +=1 
            else:
                new_queue.append(queue_node)
                
        self.high_level_planner.queue = deque(new_queue)
        for _ in range(cnt):
            start_time = datetime.datetime.now()
            execute_output = self._execute_high_level_action()
            self.step += 1
            results["selected_step"] = "high"
            time = datetime.datetime.now()-start_time
            results["time"] = time.seconds + time.microseconds/1000000
            self._log(key="results", value=results)
            self.logger_cnt += 1
            sample = self._get_empty_sample()
            sample["low_level_state"] = results['end_observation']
            sample["high_level_action"] = execute_output['node'].action.name
            sample["high_level_next_state"] = execute_output['id']
            output.append(sample)

        return output
    
    def _compute_f(self, node:int):
        if node == None:
            return 0
        current_node = self.graph_manager.low_level_graph.graph.nodes._nodes[node]
        parent_f = self._compute_f(node=current_node['parent_id'])
        
        #calculate node f
        tries = sum(current_node['tries'])
        number_of_siblings = self._get_number_of_siblings(node=node)
        node_f = (tries)^self.pc + (number_of_siblings)^self.pw + self.epslion

        return node_f + parent_f
    
    def _get_number_of_siblings(self, node:int):
        current_node = self.graph_manager.low_level_graph.graph.nodes._nodes[node]
        if current_node["parent_id"] is None:
            return 1
        num = sum(self.graph_manager.high_level_graph.graph.out_degree._nodes[current_node["parent_id"]])
        return num
    
    def run(self):
        while len(self.queue) > 0 and self.time < self.time_limit and self.step < self.max_steps:
            print("step = ", self.step)
            start_time = datetime.datetime.now()
            node = self._pop()
            #self.queue += self._widen(node=node)
            results = self._compute(node=node)
            
            if results['reward'] < 1:
                self.queue.append(node)
            elif results['goal_reached'] == True:
                self.solutions.append(node)
                results["selected_step"] = "low"
                time = datetime.datetime.now()-start_time
                results["time"] = time.seconds + time.microseconds/1000000
                self._log(key="results", value=results)
                self.logger_cnt += 1
                return results
            else:
                new_samples = self._deepen(results=results)
                for sample in new_samples:
                    self.queue.append(sample)
                
            results["selected_step"] = "low"
            time = datetime.datetime.now()-start_time
            results["time"] = time.seconds + time.microseconds/1000000
            self._log(key="results", value=results)
            self.logger_cnt += 1
                
        return results