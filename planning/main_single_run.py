import sys
sys.path.append(".")

import argparse
import os
import copy

from utils.config_utils import load_cfg, set_output_path, save_cfg, update_task

from planning.config.low_level_catalog import LOW_LEVEL_CATALOG
from planning.main import run
from planning.main_single_run_fixed_high_level_plans import run_fixed_high_level_plans

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="planning/config/planner_v0.yml")
    parser.add_argument('--low_level_config', type=str, default="stap", choices=["stap", "diffusion"])
    parser.add_argument('--output_path', type=str, default="no_git/system/no_limit_tamp")
    parser.add_argument('--p_high_level', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--system', type=str, default="tamp", choices=["tamp", "els_v2", "els_v3"])
    parser.add_argument('--max_steps', type=float, default=None)
    parser.add_argument('--number_of_high_level_plans', type=int, default=None)
    parser.add_argument('--turn_off_high_level', type=bool, default=False)
    parser.add_argument('--pddl_problem', type=str, default="planning/config/problems/0.pddl")
    parser.add_argument('--max_time', type=float, default=None)
    parser.add_argument('--stochastic_actions', action='store_true')
    parser.add_argument('--dont_check_folder', action='store_true')
    parser.add_argument('--state_estimator', required=True, type=str, choices=["p_value", "q_value"])


    args = parser.parse_args()
    main_folder = copy.copy(args.output_path)

    assert args.system in args.output_path
    
    seed = args.seed
    args.output_path = main_folder 
    args.p_high_level = seed
    output_path = os.path.join(args.output_path, str(args.p_high_level)+"_single")
    files_inside = os.listdir(output_path) if os.path.exists(output_path) else []
    if args.dont_check_folder or (len(files_inside) == 0 or not os.path.exists(os.path.join(output_path, files_inside[0],"log.pickle"))):
        args.output_path = output_path
        os.makedirs(args.output_path, exist_ok=True)
        cfg = load_cfg(config_path=args.config_path, load_as_edict=True)
        if args.max_steps is not None:
            cfg["GENERAL_PARMAS"]["max_steps"] = args.max_steps
        if args.max_time is not None:
            cfg["GENERAL_PARMAS"]["max_time"] = args.max_time
        
        #if we use constraints, we need to set the pddl problem
        if args.config_path == "planning/config/planner_v0.yml":
            if args.pddl_problem is not None:
                cfg["GENERAL_PARMAS"]["pddl_problem"] = args.pddl_problem
            cfg = update_task(cfg=cfg)
        cfg['GENERAL_PARMAS']['seed'] = seed
        low_level_config = LOW_LEVEL_CATALOG[args.low_level_config]
        cfg["LOW_LEVEL_PLANNER"] = load_cfg(config_path=low_level_config)
        cfg['LOW_LEVEL_PLANNER']['params']['seed'] = cfg['GENERAL_PARMAS']['seed']
        cfg['LOW_LEVEL_PLANNER']['params']['task_path'] = cfg['GENERAL_PARMAS']['task_path']
        cfg["GENERAL_PARMAS"]["p_high_level"] = args.p_high_level
        cfg["GENERAL_PARMAS"]["system_type"] = args.system
        cfg["GENERAL_PARMAS"]["turn_off_high_level"] = args.turn_off_high_level
        cfg["GENERAL_PARMAS"]["number_of_high_level_plans"] = args.number_of_high_level_plans
        cfg["GENERAL_PARMAS"]["stochastic_actions"] = args.stochastic_actions
        cfg.GENERAL_PARMAS.state_estimator.type = args.state_estimator

        cfg = set_output_path(cfg=cfg, output_path=args.output_path)
        save_cfg(cfg=cfg, output_folder=cfg.GENERAL_PARMAS.output_path)
        if cfg["GENERAL_PARMAS"]["turn_off_high_level"]:
            run_fixed_high_level_plans(cfg=cfg)
        else:
            run(cfg=cfg)
    else:
        print(f'path exist {output_path}')
        
