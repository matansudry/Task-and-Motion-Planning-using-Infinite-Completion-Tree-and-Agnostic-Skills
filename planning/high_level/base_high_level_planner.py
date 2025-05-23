import symbolic
from abc import ABC, abstractmethod 


class BaseHighLevelPlanner():
    def __init__(self, params:dict={}):
        self.params = params
        self.max_depth = self.params["max_depth"]
        self.timeout = self.params["timeout"]
        self.verbose = self.params["verbose"]

    @abstractmethod
    def plan(self, pddl_domain:str, pddl_problem:str):
        pass