
import datetime
import numpy as np
import networkx as nx

from planning.bandits.catalog import BANDITS_CATALOG
from planning.system.our_tamp import TAMPPlanner

class ELSv3SystemPlanner(TAMPPlanner):
    def __init__(self, cfg:dict):
        super().__init__(cfg=cfg)

        self.high_level_cn = 0
        self.high_level_jn = 1


    def _get_high_level_plan(self):
        cnt = 0
        self.high_level_cn += 1
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
                self.high_level_jn += 1
                return results
            
            if self.step > self.max_steps:
                return None
        
        return None

    def _get_high_level_option_score(self):
        depth_score = pow(self.high_level_cn/self.cfg.GENERAL_PARMAS.els.c0, self.cfg.GENERAL_PARMAS.els.pc)
        width_score = pow(self.high_level_jn/self.cfg.GENERAL_PARMAS.els.w0, self.cfg.GENERAL_PARMAS.els.pw)
        score = depth_score + width_score + self.cfg.GENERAL_PARMAS.els.epsilon

        return score

    def _get_single_score(self, state:int, plan:list):
        jn = self.graph_manager.low_level_graph.graph.nodes._nodes[state]["tries"][plan[1]] if\
             len(self.graph_manager.low_level_graph.graph.nodes._nodes[state]["tries"]) > 0 and plan[1] in self.graph_manager.low_level_graph.graph.nodes._nodes[state]["tries"]\
            else 0
        cn = 1
        depth_score = pow(cn/self.cfg.GENERAL_PARMAS.els.c0, self.cfg.GENERAL_PARMAS.els.pc)
        width_score = pow(jn/self.cfg.GENERAL_PARMAS.els.w0, self.cfg.GENERAL_PARMAS.els.pw)
        score = depth_score + width_score + self.cfg.GENERAL_PARMAS.els.epsilon

        return score

    def _get_score(self, low_level_states:list, plan:list):
        goal = low_level_states[-1]["index"]
        if goal == 0:
            plans = [[0]]
        else:
            plans = list(nx.all_simple_paths(self.graph_manager.low_level_graph.graph, source=0, target=low_level_states[0]["index"]))
        current_low_level_plan = plans[0]
        score = 0
        for i in range(len(current_low_level_plan)):
            score += self._get_single_score(state=current_low_level_plan[i], plan=plan)
        return score

    def _get_plan_score(self, plan):
        #select low level state
        selected_high_level_state = \
            self.graph_manager.high_level_graph.graph.nodes._nodes[plan[0]]["state"]
        low_level_states =\
            self.graph_manager.get_all_low_level_states_from_high_level_state(state=selected_high_level_state)

        score = self._get_score(low_level_states=low_level_states, plan=plan)

        return score, low_level_states[-1]["index"], None


    def _select_option(self, options:list) -> dict:
        """_summary_

        Args:
            options (list): _description_

        Returns:
            dict: _description_
        """
        ids = [i for i in range(len(options))]
        probs = np.array([option["score"] for option in options])
        probs /= sum(probs)
        choice_id = np.argmin(probs)
        
        if options[choice_id]["name"] == 'open_high_level':
            self.high_level_reset = True
        else:
            self.low_level_reset = True
        
        return options[choice_id]