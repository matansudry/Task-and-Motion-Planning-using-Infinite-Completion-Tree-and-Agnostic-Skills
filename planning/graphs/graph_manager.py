import networkx as nx
import numpy as np

from pyperplan_repo.pyperplan.search.searchspace import SearchNode
from planning.graphs.high_level_graph import HighLevelGraph
from planning.graphs.low_level_graph import LowLevelGraph
from planning.utils.env_utils import fix_high_level_action


class GraphManager():
    def __init__(self):
        self.high_level_graph = HighLevelGraph()
        self.low_level_graph = LowLevelGraph()
        
    def add_high_level_node(self, high_level_state, goal_reached:bool):

        id = self.high_level_graph.add_node(
            high_level_state=high_level_state,
            goal_reached=goal_reached,
        )
        
        return id
        
    def get_all_high_level_trajctories(self) -> list:
        """_summary_

        Returns:
            list: _description_
        """
        trajctories = self.high_level_graph.get_all_trajctories()
        
        return trajctories
    
    def get_low_level_plan(self):
        plan = self.low_level_graph.get_all_trajctories()[0]
        
        return plan
    
    def add_low_level_node(self, initial_high_level_state, high_level_state,\
        high_level_action, low_level_action, initial_low_level_state, low_level_state,\
        goal_reached:bool, info:dict, robot_state:np.array):
        """_summary_

        Args:
            initial_high_level_state (_type_): _description_
            end_high_level_state (_type_): _description_
            high_level_action (_type_): _description_
            low_level_action (_type_): _description_
            initial_low_level_state (_type_): _description_
            end_low_level_state (_type_): _description_
        """
        high_level_node_id = self.high_level_graph.get_parent_id(parent_state=high_level_state)
        high_level_action_id = self.high_level_graph.get_edge_id(
            state=initial_high_level_state,
            action=high_level_action
        )
        self.low_level_graph.add_node(
            parent_low_level_state=initial_low_level_state,
            low_level_state=low_level_state,
            high_level_node_id=high_level_node_id,
            high_level_action_id=high_level_action_id,
            goal_reached=goal_reached,
            low_level_action=low_level_action,
            info=info,
            robot_state=robot_state,
        )
        
        self.low_level_graph.node_complete(
            parent_low_level_state=initial_low_level_state,
            )  
        
    def get_all_low_level_states_from_high_level_state(self, state):
        high_level_indexes = []
        for high_level_node_index in self.high_level_graph.graph.nodes._nodes:
            if self.high_level_graph.graph.nodes._nodes[high_level_node_index]['state'] == state:
                high_level_indexes.append(high_level_node_index)
        
        low_level_states = []
        for high_level_index in high_level_indexes:
            for node_index in self.low_level_graph.graph.nodes._nodes:
                if self.low_level_graph.graph.nodes._nodes[node_index]['high_level_node_id'] ==\
                    high_level_index:
                    low_level_states.append(self.low_level_graph.graph.nodes._nodes[node_index])
                    
        return low_level_states
            