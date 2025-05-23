import sys
sys.path.append(".")

import argparse
import tqdm
import os
import copy

from utils.config_utils import load_cfg, set_output_path, save_cfg
from utils.general_utils import save_json

from planning.system.catalog import SYSTEM_CATALOG
from planning.config.low_level_catalog import LOW_LEVEL_CATALOG
import datetime
   
def run(cfg:dict):
    system = SYSTEM_CATALOG[cfg.GENERAL_PARMAS.system_type](cfg=cfg)
    total_time = system.run()
    system.save(time=total_time)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="planning/config/planner_v0.yml")
    parser.add_argument('--low_level_config', type=str, default="stap", choices=["stap", "diffusion"])
    parser.add_argument('--output_path', type=str, default="no_git/system/playground_2")
    parser.add_argument('--p_high_level', type=float, default=0.3)

    print("need to fix code to single run")
    raise
    
    args = parser.parse_args()
    main_folder = copy.copy(args.output_path)
    
    for p in [args.p_high_level]:
        args.output_path = main_folder
        args.p_high_level = p
        args.output_path = os.path.join(args.output_path, str(p)[-1])
        os.makedirs(args.output_path, exist_ok=True)
        for seed in range(100):
            output_path = os.path.join(args.output_path, str(seed))
            os.makedirs(output_path, exist_ok=True)
            cfg = load_cfg(config_path=args.config_path, load_as_edict=True)
            cfg['GENERAL_PARMAS']['seed'] = seed
            low_level_config = LOW_LEVEL_CATALOG[args.low_level_config]
            cfg["LOW_LEVEL_PLANNER"] = load_cfg(config_path=low_level_config)
            cfg['LOW_LEVEL_PLANNER']['params']['task_path'] = cfg['GENERAL_PARMAS']['task_path']
            cfg['LOW_LEVEL_PLANNER']['params']['seed'] = cfg['GENERAL_PARMAS']['seed']
            cfg["GENERAL_PARMAS"]["p_high_level"] = args.p_high_level
            
            cfg = set_output_path(cfg=cfg, output_path=output_path)
            save_cfg(cfg=cfg, output_folder=cfg.GENERAL_PARMAS.output_path)
            
            run(cfg=cfg)
    
