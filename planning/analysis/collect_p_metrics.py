import sys
sys.path.append(".")

import argparse
import os
import pandas as pd
import numpy as np
from utils.general_utils import load_pickle, save_pickle
import os, shutil
import tqdm
from planning.analysis.analyser import logger_to_df

def create_plot(p_value:list, success:list, path:str):
    import matplotlib.pyplot as plt

    temp = np.array(p_value)
    bool_list = [True if item else False for item in success]
    print("good avg = ", temp[bool_list].mean())
    neg_bool_list = [False if item else True for item in success]
    print("bad avg = ", temp[neg_bool_list].mean())
    
    x = np.arange(len(p_value))
    colormap = np.array(['r', 'g'])
    plt.title(f'good avg = {temp[bool_list].mean()} and bad avg = {temp[neg_bool_list].mean()}')
    plt.scatter(x, np.array(p_value), s=1, c=colormap[np.array(success).astype(np.int32)])
    plt.savefig(path)
    plt.cla()

if __name__ == "__main__":
    states = []
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str,\
         default="no_git/system/eval_v4_els_v2_els_easy_1200")
    args = parser.parse_args()
    for algo in ["p", "q"]:
        for diffucalt in ["easy", "mid", "hard"]:
            main_folder = f'no_git/system/eval_v4_{algo}_v2_{algo}_{diffucalt}_1200'
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
            success_samples = []
            estimator_samples = []
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
                        #if not os.path.isfile(df_path):
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
                            'selected_step', 'end_observation',\
                                'terminated', 'truncated'])
                        df.to_pickle(df_path)
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
                        
                        df = df[~df["first_action_score"].isnull()]
                        success_samples += df["reward"].tolist()
                        estimator_samples += df["first_action_score"].tolist()
            create_plot(
                p_value=estimator_samples,
                success=success_samples,
                path=f'{algo}_{diffucalt}.png'
            )