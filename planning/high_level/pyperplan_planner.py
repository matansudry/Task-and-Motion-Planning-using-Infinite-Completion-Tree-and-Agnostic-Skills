from pyperplan_repo.pyperplan.planner import (
    HEURISTICS,
    SEARCHES,
)

from collections import deque
from pyperplan_repo.pyperplan.planner import _parse, _ground
from pyperplan_repo.pyperplan.heuristics.relaxation import hFFHeuristic
from pyperplan_repo.pyperplan.search.searchspace import make_root_node, make_child_node
from planning.high_level.base_high_level_planner import BaseHighLevelPlanner



class PyperplanPlanner(BaseHighLevelPlanner):
    def __init__(self, params:dict={}):
        super().__init__(params=params)
        self.last_plan_found_step = 0
        self.how_much_steps_in_last_plan_found = None
        self.id = 0
        
    def _set_search_method(self, search_method:str):
        self.search_method = search_method
        self.search = SEARCHES[search_method]
        
    def _set_heuristic(self, heuristic_method:str):
        if self.search_method in ["bfs", "ids", "sat"]:
            self.heuristic = None
        else:
            
            self.heuristic = HEURISTICS[heuristic_method]
        self.use_preferred_ops = heuristic_method == "hffpo"
    
    def _prepare_task(self, pddl_domain:str, pddl_problem:str):
        problem = _parse(pddl_domain, pddl_problem)
        task = _ground(problem)
        
        return task
    
    def init_search(self, pddl_domain:str, pddl_problem:str, search_method:str, heuristic_method:str,\
        ):
        self._set_search_method(search_method=search_method)
        self._set_heuristic(heuristic_method=heuristic_method)
        self.task = self._prepare_task(pddl_domain=pddl_domain, pddl_problem=pddl_problem)
        
        #get heuristic
        heuristic = None
        if not self.heuristic is None:
            heuristic = self.heuristic(self.task)

        #check heuristic is heuristics.hFFHeuristic, if not change use_preferred_ops to False
        if not isinstance(heuristic, hFFHeuristic):
            self.use_preferred_ops = False
            
        
        self.root = make_root_node(self.task.initial_state)
        self.queue = deque()
        self.queue.append(self.root)
        self.id += 1
        # set storing the explored nodes, used for duplicate detection
        self.closed = {self.task.initial_state}
        self.solutions = list()
        #self.graph_manager = GraphManager()
        #self.graph_manager.add_node(self.root)
    
    def get_all_plans(self) -> list:
        """
        get all high level plans

        Returns:
            list: plans
        """

        for _ in range(10000):
            success = self.run_one_step()
            if success == False:
                break
        #plans_graph = self.graph_manager.get_all_trajctories()
        plans_planner = self.solutions
        
        return plans_planner #plans_graph, plans_planner
    
    def _get_output(self):
        output = {
            "queue_isnt_empty": True,
            "found_goal": False,
            "node": None,
            "plan": [],
        }
        
        return output
    
    def get_estimate_how_much_nodes_left(self):
        if self.how_much_steps_in_last_plan_found is None:
            return 1
        steps_left = (self.how_much_steps_in_last_plan_found + self.last_plan_found_step) - self.last_id
        return steps_left
    
    def update_search_effort(self, id:int):
        self.how_much_steps_in_last_plan_found = id - self.last_plan_found_step
        self.last_plan_found_step = id
    
    def run_one_step(self):
        output = self._get_output()
        
        if len(self.queue) == 0:
            output["queue_isnt_empty"] = False
            return output
        
        node = self.queue.popleft()
        self.last_id = node.id
        if node.g > self.max_depth:
            return output
        output["node"] = node
        # exploring the node or if it is a goal node extracting the plan
        if self.task.goal_reached(node.state):
            solution = node.extract_solution()
            self.solutions.append(solution)
            output["found_goal"] = True
            self.update_search_effort(id = output['node'].id)
            output["plan"] = solution
            return output
        for operator, successor_state in self.task.get_successor_states(node.state):
            # duplicate detection
            if successor_state not in self.closed:
                self.queue.append(
                    make_child_node(node, operator, successor_state, id=self.id)
                )
                self.id +=1
                # remember the successor state
                self.closed.add(successor_state) 
        return output

