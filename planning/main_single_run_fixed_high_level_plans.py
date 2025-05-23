import sys
sys.path.append(".")

import argparse
import os
import copy
import datetime

from utils.config_utils import load_cfg, set_output_path, save_cfg

from planning.system.catalog import SYSTEM_CATALOG
from planning.config.low_level_catalog import LOW_LEVEL_CATALOG

def run_fixed_high_level_plans(cfg:dict):
    system = SYSTEM_CATALOG[cfg.GENERAL_PARMAS.system_type](cfg=cfg)
    for _ in range(cfg["GENERAL_PARMAS"]["number_of_high_level_plans"]):
        system._roll_out_high_level()
    total_time = system.run()
    system.save(time=total_time)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="planning/config/planner_v0.yml")
    parser.add_argument('--low_level_config', type=str, default="stap", choices=["stap", "diffusion"])
    parser.add_argument('--output_path', type=str, default="no_git/system/fixed_els_v2")
    parser.add_argument('--p_high_level', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--system', type=str, default="els_v2", choices=["tamp", "els_v2"])
    
    args = parser.parse_args()
    main_folder = copy.copy(args.output_path)
    
    assert args.system in args.output_path
    
    seed = args.seed
    args.output_path = main_folder 
    args.p_high_level = seed
    output_path = os.path.join(args.output_path, str(args.p_high_level)+"_single")
    if not os.path.exists(output_path):
        args.output_path = output_path
        os.makedirs(args.output_path, exist_ok=True)
        cfg = load_cfg(config_path=args.config_path, load_as_edict=True)
        cfg['GENERAL_PARMAS']['seed'] = seed
        low_level_config = LOW_LEVEL_CATALOG[args.low_level_config]
        cfg["LOW_LEVEL_PLANNER"] = load_cfg(config_path=low_level_config)
        cfg['LOW_LEVEL_PLANNER']['params']['task_path'] = cfg['GENERAL_PARMAS']['task_path']
        cfg['LOW_LEVEL_PLANNER']['params']['seed'] = cfg['GENERAL_PARMAS']['seed']
        cfg["GENERAL_PARMAS"]["p_high_level"] = args.p_high_level
        
        cfg["GENERAL_PARMAS"]["system_type"] = args.system
        cfg = set_output_path(cfg=cfg, output_path=args.output_path)
        save_cfg(cfg=cfg, output_folder=cfg.GENERAL_PARMAS.output_path)
        run_fixed_high_level_plans(cfg=cfg)
        
