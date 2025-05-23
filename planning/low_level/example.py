import sys
sys.path.append(".")

from planning.high_level.catalog import HIGH_LEVEL_PLANNERS

if __name__ == "__main__":
    
    pddl_domain = "configs/pybullet/envs/official/sim_domains/constrained_packing/tamp0_domain_new.pddl"
    pddl_problem = "configs/pybullet/envs/official/sim_domains/constrained_packing/tamp0_problem_new.pddl" #"configs/pybullet/envs/official/sim_domains/constrained_packing/tamp0_problem.pddl"
    search_method = "bfs"
    heuristic_name = None
    params = {
        "max_depth": 4,
        "timeout": 10, 
        "verbose": False,
    }
    
    planner = HIGH_LEVEL_PLANNERS["pyperplan"](params=params)
    plans = planner.init_search(
        pddl_domain=pddl_domain,
        pddl_problem=pddl_problem,
        search_method=search_method,
        heuristic_method=heuristic_name
    )
    for _ in range(10000):
        success = planner.run_one_step()
        if success == False:
            break
    plans_graph = planner.graph_manager.get_all_trajctories()
    plans_planner = planner.solutions