import networkx as nx
import numpy as np

from pyperplan_repo.pyperplan.search.searchspace import SearchNode

class HighLevelGraph():
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

    def add_node(self, high_level_state:SearchNode, goal_reached:bool=False) -> int:
        """_summary_

        Args:
            node (SearchNode): _description_
            goal_reached (bool, optional): _description_. Defaults to False.

        Returns:
            int: _description_
        """
        parent_state = high_level_state.parent.state if high_level_state.parent is not None else None
        parent_id = self.get_parent_id(parent_state=parent_state)
        node_attribute_dict = {
            "action": high_level_state.action,
            "g": high_level_state.g,
            "parent": parent_id,
            "state": high_level_state.state,
            "goal_reached": goal_reached,
        }
        
        self.graph.add_nodes_from([
            (self.id, node_attribute_dict),
        ])
        
        if parent_id is not None:
            edge_attibutes = {
                "high_level_action": high_level_state.action.name,
            }
            self.add_edge(
                from_id=parent_id, 
                to_id=self.id, 
                attribute=edge_attibutes,
            )
        
        self.id += 1

        return self.id -1

    def get_edge_id(self, state, action):
        if action is None:
            return None
        state_id = self.get_parent_id(parent_state=state)
        edges_options = self.graph.edges._adjdict[state_id]
        for option in edges_options:
            if edges_options[option]['high_level_action'] == action:
                return option
            
        raise


    def get_parent_id(self, parent_state) -> int:
        """_summary_

        Args:
            parent_state (dict): _description_

        Returns:
            int: _description_
        """
        if parent_state is None:
            return None
        for node in self.graph.nodes():
            node_attributes = self.graph.nodes._nodes[node]
            if node_attributes['state'] == parent_state:
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
        
        while node_attributes['parent']!= None:
            node_attributes = self.graph.nodes._nodes[node_attributes['parent']]
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