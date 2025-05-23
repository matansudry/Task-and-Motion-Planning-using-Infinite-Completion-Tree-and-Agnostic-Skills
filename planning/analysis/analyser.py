import sys
sys.path.append(".")

import argparse
import os
import pandas as pd
import numpy as np
from utils.general_utils import load_pickle, save_pickle
import os, shutil
import tqdm

def logger_to_df(logger:dict):
    output = {
        "index" :[],
        "queue_isnt_empty" :[],
        "found_goal" :[],
        "node" :[],
        "plan" :[],
        "type" :[],
        "selected_step" :[],
        "time" :[],
        "end_observation" :[],
        "reward" :[],
        "terminated" :[],
        "truncated" :[],
        "info" :[],
        "goal_reached" :[],
        "first_action_score": [],
    }
    for log in tqdm.tqdm(logger['results']):
        output["index"].append(log["index"])
        queue_isnt_empty = log["queue_isnt_empty"] if "queue_isnt_empty" in log else None
        output["queue_isnt_empty"].append(queue_isnt_empty)
        found_goal = log["found_goal"] if "found_goal" in log else None
        output["found_goal"].append(found_goal)
        node = log["node"] if "node" in log else None
        output["node"].append(node)
        plan = log["plan"] if "plan" in log else None
        output["plan"].append(plan)
        output["type"].append(log["type"])
        output["selected_step"].append(log["selected_step"])
        output["time"].append(log["time"])
        end_observation = log["end_observation"] if "end_observation" in log else None
        output["end_observation"].append(end_observation)
        reward = log["reward"] if "reward" in log else None
        output["reward"].append(reward)
        terminated = log["terminated"] if "terminated" in log else None
        output["terminated"].append(terminated)
        truncated = log["truncated"] if "truncated" in log else None
        output["truncated"].append(truncated)
        info = log["info"] if "info" in log else None
        output["info"].append(info)
        goal_reached = log["goal_reached"] if "goal_reached" in log else None
        output["goal_reached"].append(goal_reached)
        first_action_score = log["first_action_score"] if "first_action_score" in log else None
        output["first_action_score"].append(first_action_score)
        
    df = pd.DataFrame.from_dict(output)
    return df

if __name__ == "__main__":
    states = []
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str,\
         default="no_git/system/eval_v4_els_v2_els_easy_1200")
    args = parser.parse_args()
    for diffucalt in ["easy", "mid", "hard"]:
        main_folder = f'no_git/system/eval_v4_els_v3_els_{diffucalt}_1200'
        hard_states = []
        output = {
            "success": [],
            "low_level_steps": [],
            "high_level_steps": [],
            "high_level_plans": [],
            "time": [],
            "low_level_time": [],
            "high_level_time": [],
            "total_time": []
        }
        for rate in tqdm.tqdm(list(os.listdir(main_folder))):
            rate_folder = os.path.join(main_folder, rate)
            runs = list(os.listdir(rate_folder))
            if len(runs) > 1 or len(runs)==0:
                print(rate_folder)
                #shutil.rmtree(rate_folder)
                continue
            assert len(runs) == 1, print(rate_folder)
            for exp in runs:
                if os.path.isfile(os.path.join(rate_folder, exp,"log.pickle")):
                    log_path = os.path.join(rate_folder, exp,"log.pickle")
                    df_path = os.path.join(rate_folder, exp, "df.pkl")
                    if not os.path.isfile(df_path):
                        try:
                            logger = load_pickle(log_path)
                        except:
                            print(log_path)
                        time = logger['total_time'] if "total_time" in logger else 0
                        if time > logger['allowed_time'] or len(logger['results']) > 10000000000:
                            output["success"].append(False)
                            continue
                        df = logger_to_df(logger=logger)
                        df = df.drop(columns=[
                            "node", "info", 'queue_isnt_empty', 'plan',\
                            'selected_step', 'end_observation', 'reward',\
                                'terminated', 'truncated'])
                        df.to_pickle(df_path)
                    else:
                        df = pd.read_pickle(df_path)  
    
                    #if time > 900:
                    #    continue
                        #print("time = ", time)
                    #df = df.iloc[:1000]
                    #if df.shape[0] > 1000:
                        #print(df.shape[0])
                    #load graph
                    graph_path = os.path.join(rate_folder, exp, "low_level_grpah.gpickle")
                    import networkx as nx
                    temp = load_pickle(graph_path)
                    state = np.array(temp.graph.nodes._nodes[0]['low_level_state'].reshape(-1))
                    same_state = False
                    for target_state in states:
                        if np.linalg.norm(state-target_state) < 0.0001:
                            same_state = True
                            break
                    if same_state:
                        continue   
                    states.append(state)
                    
                    
                    success = (df["goal_reached"]==True).sum() > 0
                    output["success"].append(success)
                    if success:
                        steps = dict(df["type"].value_counts())
                        output["high_level_steps"].append(steps[('high_level',)])
                        output["low_level_steps"].append(steps['low_level'])
                        output["high_level_plans"].append(df["found_goal"].sum())
                        output["time"].append(df["time"].sum())
                        low_level_df = df.query('type == "low_level"')
                        high_level_df = df.query('type != "low_level"')
                        output["low_level_time"].append(low_level_df["time"].sum())
                        output["high_level_time"].append(high_level_df["time"].sum())
                        #output["total_time"].append(time)
                    else:
                        seed = rate.split("_")[0]
                        hard_states.append(seed)
                        #print()
                else:
                    print("matan = ", os.path.join(rate_folder, exp))
                    #shutil.rmtree(os.path.join(rate_folder, exp))
                        
                        
        temp=1
        
        print("diffucalt =", diffucalt)
        print("tries = ", len(output["success"]))
        print("success = ", np.sum(output["success"]))
        print("high_level_steps = ", np.mean(output['high_level_steps']))
        print("low_level_steps = ", np.mean(output['low_level_steps']))
        print("high_level_plans = ", np.mean(output['high_level_plans']))
        print("time = ", np.mean(output['time']))
        print("high_level_time = ", np.mean(output['high_level_time']))
        print("low_level_time = ", np.mean(output['low_level_time']))
        #print("total_time = ", np.mean(output['total_time']))
        temp=1
        
        
        """
        tries =  100
        success =  31
        high_level_steps =  58019.67741935484
        low_level_steps =  371.5806451612903
        high_level_plans =  1.0
        time =  0.10106874193548389
        high_level_time =  0.10106874193548389
        low_level_time =  0.0
        """