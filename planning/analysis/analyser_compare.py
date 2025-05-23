import sys
sys.path.append(".")

import argparse
import tqdm
import os
import pandas as pd
import numpy as np
from utils.general_utils import load_pickle
from planning.analysis.analyser import logger_to_df


if __name__ == "__main__":
    for diffucalt in ["mid", "hard"]: #["easy", "mid", "hard"]:
        states = {}
    
        algo_one = "els"
        one_name = f'eval_v4_{algo_one}_v2_{algo_one}_{diffucalt}_1200' #f'eval_v2_{algo_one}_{diffucalt}_1200'
        folder_one_path = f'no_git/system/{one_name}'
        
        #two_name = f'tamp_eval_tamp_with_q_and_p_{diffucalt}_1200_q_value_actions_exp_with_first_action_score'
        algo_two = "els"
        two_name = f'eval_v4_{algo_two}_v3_{algo_two}_{diffucalt}_1200'
        folder_two_path = f'no_git/system/{two_name}'
        
        #load all states
        names = runs = list(os.listdir(folder_one_path))
        used_stated = []
        for name in tqdm.tqdm(names):
            run_one = os.path.join(folder_one_path, name)
            one_runs = os.listdir(run_one)
            assert len(one_runs) == 1, f"run_one = {run_one}"
            run_one = os.path.join(run_one, one_runs[0])
            if os.path.isfile(os.path.join(run_one, "log.pickle")):
                logger_one = load_pickle(os.path.join(run_one, "log.pickle"))
                one_time = logger_one['total_time']
                if os.path.isfile(os.path.join(run_one, "df.pickle")):
                    df_one = pd.read_pickle(os.path.join(run_one, "df.pickle"))
                else:
                    df_one = logger_to_df(logger=logger_one)
                    df_one = df_one.drop(columns=[
                        "node", "info", 'queue_isnt_empty', 'plan',\
                        'selected_step', 'end_observation', 'reward',\
                            'terminated', 'truncated'])
                    df_one.to_pickle(os.path.join(run_one, "df.pickle"))
                graph_path = os.path.join(run_one, "low_level_grpah.gpickle")
                import networkx as nx
                temp = load_pickle(graph_path)
                np_state = np.array(temp.graph.nodes._nodes[0]['low_level_state'].reshape(-1))
                state = tuple(np_state.tolist())
                
                same_state = False
                for target_state in used_stated:
                    if np.linalg.norm(np_state-target_state) < 0.0001:
                        same_state = True
                        break
                if same_state:
                    continue   
                used_stated.append(np_state)
                
                if state in states:
                    continue
                success = (df_one["goal_reached"]==True).sum() > 0
                if (success and one_time >logger_one['allowed_time']) or (not success and one_time < logger_one['allowed_time']):
                    raise
                states[state] = {
                    "one_time": one_time
                }
                
                
        names = runs = list(os.listdir(folder_two_path))
        used_stated = []
        cnt_success = 0
        for name in tqdm.tqdm(names):
            run_two = os.path.join(folder_two_path, name)
            two_runs = os.listdir(run_two)
            assert len(two_runs) == 1
            run_two = os.path.join(run_two, two_runs[0])
            if os.path.isfile(os.path.join(run_two, "log.pickle")):
                try:
                    logger_two = load_pickle(os.path.join(run_two, "log.pickle"))
                except:
                    continue
                two_time = logger_two['total_time']
                if os.path.isfile(os.path.join(run_two, "df.pickle")):
                    df_two = pd.read_pickle(os.path.join(run_two, "df.pickle"))
                else:
                    df_two = logger_to_df(logger=logger_two)
                    df_two = df_two.drop(columns=[
                        "node", "info", 'queue_isnt_empty', 'plan',\
                        'selected_step', 'end_observation', 'reward',\
                            'terminated', 'truncated'])
                    df_two.to_pickle(os.path.join(run_two, "df.pickle"))
                graph_path = os.path.join(run_two, "low_level_grpah.gpickle")
                import networkx as nx
                temp = load_pickle(graph_path)
                np_state = np.array(temp.graph.nodes._nodes[0]['low_level_state'].reshape(-1))
                state = tuple(np_state.tolist())                
                same_state = False
                for target_state in used_stated:
                    if np.linalg.norm(np_state-target_state) < 0.0001:
                        same_state = True
                        break
                if same_state:
                    continue   
                used_stated.append(np_state)
                
                if state not in states:
                    continue
                success = (df_two["goal_reached"]==True).sum() > 0
                cnt_success += success
                if (success and two_time >logger_two['allowed_time']) or (not success and two_time < logger_two['allowed_time']):
                    raise
                states[state]["two_time"] = two_time
        print(f"cnt_success = {cnt_success}")

        output = {
            #"run_name": [],
            f'{one_name}_time': [],
            f'{one_name}_success': [],
            f'{two_name}_time': [],
            f'{two_name}_success': [],
        }
        for key in states:
            try:
                output[f'{one_name}_time'].append(states[key]["one_time"])
                output[f'{one_name}_success'].append(1 if states[key]["one_time"]<logger_one['allowed_time'] else 0)
                output[f'{two_name}_time'].append(states[key]["two_time"])
                output[f'{two_name}_success'].append(1 if states[key]["two_time"]<logger_one['allowed_time'] else 0)
            except:
                continue
        df = pd.DataFrame.from_dict(output)
        df.to_csv(f'out_{algo_one}_vs_{algo_two}_{diffucalt}_v3.csv')

        """
        for name in names:
            run_one = os.path.join(folder_one_path, name)
            one_runs = os.listdir(run_one)
            assert len(one_runs) == 1
            run_one = os.path.join(run_one, one_runs[0])
            
            run_two = os.path.join(folder_two_path, name)
            two_runs = os.listdir(run_two)
            assert len(two_runs) == 1
            run_two = os.path.join(run_two, two_runs[0])
            
            if os.path.isfile(os.path.join(run_one, "log.pickle")) and\
                os.path.isfile(os.path.join(run_two, "log.pickle")):
                
                logger_one = load_pickle(os.path.join(run_one, "log.pickle"))
                one_time = logger_one['total_time']
                df_one = logger_to_df(logger=logger_one)
                logger_two = load_pickle(os.path.join(run_two, "log.pickle"))
                df_two = logger_to_df(logger=logger_two)
                two_time = logger_two['total_time']
                if (df_one["goal_reached"]==True).sum() > 0 and (df_two["goal_reached"]==True).sum() > 0:
                    output["run_name"].append(name)
                    output[f'{one_name}_time'].append(one_time)
                    output[f'{two_name}_time'].append(two_time)
        df = pd.DataFrame.from_dict(output)
        df.to_csv(f'out_fixed_{diffucalt}.csv')
        """