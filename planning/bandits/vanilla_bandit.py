

class VanillaBandit():
    def __init__(self, params:dict={}):
        self.params = params

    def score(self, visits:int):
        score = 1 / (1+visits)
        return score