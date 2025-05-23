from abc import ABC, abstractmethod 


class BaseLowLevelPlanner():
    def __init__(self, params:dict={}):
        self.params = params

    @abstractmethod
    def plan(self):
        pass