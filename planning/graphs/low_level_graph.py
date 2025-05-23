import networkx as nx
import numpy as np

from pyperplan_repo.pyperplan.search.searchspace import SearchNode

class LowLevelGraph():
    def __init__(self):
        self.graph = nx.DiGraph()
        self.id = 0

    def add_edge(self, from_id:int, to_id:int, attribute:dict):
        """_summary_

        Args:
            from_id (int): _description_
            to_id (int): _description_
            attribute (dict): _description_
        """
        self.graph.add_edges_from([
            (from_id, to_id, attribute)
        ])

    def node_complete(self, parent_low_level_state:np.ndarray):
        parent_id = self.get_parent_id(parent_state=parent_low_level_state)
        if parent_id is not None:
            self.graph.nodes._nodes[parent_id]['IsComplete'] = True

    def add_node(self, parent_low_level_state:np.ndarray, low_level_state:np.ndarray,\
        high_level_node_id:int , high_level_action_id:int, info:dict, robot_state,\
        goal_reached:bool=False, low_level_action:np.ndarray=None) -> int:
        """_summary_

        Args:
            node (SearchNode): _description_
            goal_reached (bool, optional): _description_. Defaults to False.

        Returns:
            int: _description_
        """
        parent_id = self.get_parent_id(parent_state=parent_low_level_state)
        node_attribute_dict = {
            "high_level_node_id": high_level_node_id,
            "goal_reached": goal_reached,
            "low_level_state": low_level_state,
            "parent_id": parent_id,
            "info": info,
            "robot_state": robot_state,
            "tries": {},
            "IsComplete": False,
            "index": self.id,
        }
        
        self.graph.add_nodes_from([
            (self.id, node_attribute_dict),
        ])
        
        if low_level_action is not None:
            edge_attibutes = {
                "high_level_action_id": high_level_action_id,
                "low_level_action": low_level_action,
            }
            self.add_edge(
                from_id=parent_id, 
                to_id=self.id, 
                attribute=edge_attibutes,
            )
        
        self.id += 1
        
        

        return self.id -1

    def get_parent_id(self, parent_state:dict) -> int:
        """_summary_

        Args:
            parent_state (dict): _description_

        Returns:
            int: _description_
        """
        for node in self.graph.nodes():
            node_attributes = self.graph.nodes._nodes[node]
            if np.all(node_attributes['low_level_state'] == parent_state):
                return node
            
    def get_trajctory(self, node:int) -> list:
        """_summary_

        Args:
            node (int): _description_

        Returns:
            list: _description_
        """
        node_attributes = self.graph.nodes._nodes[node]
        path = [node_attributes]
        
        while node_attributes['parent_id'] != None:
            node_attributes = self.graph.nodes._nodes[node_attributes['parent_id']]
            path.append(node_attributes)
            
        path.reverse()
        return path
        
    def get_all_trajctories(self) -> list:
        """_summary_

        Returns:
            list: _description_
        """
        trajctories = []
        for node in self.graph.nodes():
            node_attributes = self.graph.nodes._nodes[node]
            if not node_attributes['goal_reached']:
                continue
            trajctories.append(self.get_trajctory(node=node))
            
        return trajctories

    def get_id_from_state(self, state):
        for node in self.graph.nodes._nodes:
            if np.all(self.graph.nodes._nodes[node]['low_level_state'] == state):
                return node