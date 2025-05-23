import symbolic

from planning.high_level.base_high_level_planner import BaseHighLevelPlanner

class BFSHighLevelPlanner(BaseHighLevelPlanner):
    def __init__(self, params:dict={}):
        super().__init__(params=params)
        
    def _load_pddl_problem(self, pddl_domain:str, pddl_problem:str):
        self.pddl = symbolic.Pddl(pddl_domain, pddl_problem)
        
    def _load_planner(self):
        self.planner = symbolic.Planner(self.pddl, self.pddl.initial_state)
        
    def plan(self, pddl_domain:str, pddl_problem:str):
        self._load_pddl_problem(
            pddl_domain=pddl_domain,
            pddl_problem=pddl_problem,
        )
        self._load_planner()
        bfs = symbolic.BreadthFirstSearch(
            self.planner.root,
            max_depth=self.max_depth,
            timeout=self.timeout,
            verbose=self.verbose
        )
        plans = []
        for plan in bfs:
            plans.append(plan)
        
        return plans
        
"""
python scripts/eval/eval_tamp.py
--planner-config configs/pybullet/planners/ablation/policy_cem.yaml
--env-config configs/pybullet/envs/official/sim_domains/hook_reach/tamp0.yaml
--policy-checkpoints
    models/agents_rl/pick/official_model.pt
    models/agents_rl/place/official_model.pt
    models/agents_rl/pull/official_model.pt
    models/agents_rl/push/official_model.pt
--dynamics-checkpoint
    models/dynamics_rl/pick_place_pull_push_dynamics/official_model.pt 
--seed 0
--pddl-domain configs/pybullet/envs/official/sim_domains/hook_reach/tamp0_domain.pddl
--pddl-problem configs/pybullet/envs/official/sim_domains/hook_reach/tamp0_problem.pddl
--max-depth 4
--timeout 10
--closed-loop 1
--num-eval 100
--path plots/tamp/hook_reach/tamp0
--verbose 0
""" 