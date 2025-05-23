
import datetime

from planning.system.base_system import BaseSystemPlanner
from planning.utils.env_utils import fix_high_level_action


class ELSv2SystemPlanner(BaseSystemPlanner):
    def __init__(self, cfg:dict):
        super().__init__(cfg=cfg)
        
        self.queue = []
        self.plans = []
        self.turn_off_high_level = self.cfg.GENERAL_PARMAS.turn_off_high_level
    
    def _roll_out_high_level(self):
        cnt = 0
        while True and cnt < self.max_high_level_steps:
            start_time = datetime.datetime.now()
            results = self._execute_high_level_action()
            self.step +=1
            cnt += 1
            #logging
            results["selected_step"] = "high"
            time = datetime.datetime.now()-start_time
            results["time"] = time.seconds + time.microseconds/1000000
            self._log(key="results", value=results)
            self.logger_cnt += 1
            if len(results['plan']) > 0:
                converted_plan = self._convert_plan(plan=results['plan'])
                self.plans.append(converted_plan)
                return results
        return results
    
    def _convert_plan(self, plan:list):
        current_id = 0
        new_plan = [current_id]
        for action in plan:
            edges = self.graph_manager.high_level_graph.graph.edges._adjdict[current_id]
            for edge in edges:
                if edges[edge]['high_level_action'] == action.name:
                    current_id = edge
                    new_plan.append(current_id)
                
        assert len(new_plan) == len(plan) + 1
        
        return new_plan
                    
    def _execute_low_level_action(self, start_high_level_state_id:int,
        end_high_level_state_id:int):
        
        start_high_level_state =\
            self.graph_manager.high_level_graph.graph.nodes._nodes[start_high_level_state_id]['state']
        end_high_level_state =\
            self.graph_manager.high_level_graph.graph.nodes._nodes[end_high_level_state_id]['state']
        goal_reached = self.graph_manager.high_level_graph.graph.nodes._nodes[end_high_level_state_id]['goal_reached']
        
        high_level_action_id = end_high_level_state_id
        high_level_action =\
            self.graph_manager.high_level_graph.graph.edges._adjdict[start_high_level_state_id][end_high_level_state_id]['high_level_action']
        high_level_action = fix_high_level_action(
            high_level_action=high_level_action
        )
        
        #select low level state
        observation, info, robot_state, low_level_state_id =\
            self._select_low_level_state(high_level_state=start_high_level_state)
            
        if observation is None:
            return

        self._increase_node_count(
            node_id=low_level_state_id,
            action_id=high_level_action_id,
            )
        self.step +=1 
        
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
                initial_high_level_state=start_high_level_state,
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
                    
    def run(self):
        total_time = 0
        start_time = datetime.datetime.now()
        while self.step < self.max_steps:
            #print(total_time)
            #find new high level plan
            if not self.turn_off_high_level:
                results = self._roll_out_high_level()
            
            #iterate plan after plan
            for plan in self.plans:
                for index in range(len(plan)-1, 0, -1):
                    time = datetime.datetime.now()-start_time
                    total_time = time.seconds + time.microseconds/1000000
                    
                    if total_time > self.cfg.GENERAL_PARMAS.max_time:
                        return total_time
                    
                    start_high_level_state_id = plan[index-1]
                    end_high_level_state_id = plan[index]
                    results = self._execute_low_level_action(
                        start_high_level_state_id=start_high_level_state_id,
                        end_high_level_state_id=end_high_level_state_id,
                    )
                    if results is not None:
                        #logging
                        results["selected_step"] = "low"
                        results["time"] = 0
                        self._log(key="results", value=results)
                        self.logger_cnt += 1
                        if results["goal_reached"]:
                            return total_time