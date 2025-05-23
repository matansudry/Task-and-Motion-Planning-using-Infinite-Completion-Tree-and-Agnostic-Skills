import sys
sys.path.append(".")

import tqdm
import argparse
import os
import pandas as pd
import numpy as np
from utils.general_utils import load_pickle
from planning.analyser import logger_to_df
import matplotlib.pyplot as plt

def get_exp_times(current_exp:str):
    #iterate on folder one
    runs = list(os.listdir(current_exp))
    current_exp_times = []
    for run in tqdm.tqdm(runs):
        run_one = os.path.join(current_exp, run)
        one_runs = os.listdir(run_one)
        assert len(one_runs) == 1
        run_one_path = os.path.join(run_one, one_runs[0])
        
        log_path = os.path.join(run_one_path,"log.pickle")
        df_path = os.path.join(run_one_path, "df.pkl")
        if os.path.isfile(df_path):
            df = pd.read_pickle(df_path)  
            if (df["goal_reached"]==True).sum() > 0:
                logger = load_pickle(log_path)
                current_exp_times.append(logger["total_time"])
    
    return current_exp_times

if __name__ == "__main__":
    difficult = "mid"
    main_folder = "no_git/system/"
    one_name = f'els_v2_one_tower_{difficult}_1200'
    two_name = f'tamp_one_tower_{difficult}_1200'
    index = 0
    max_time = 1200
    
    colors = ['tab:blue','tab:orange','tab:green','tab:red',
              'tab:purple','tab:brown','tab:pink','tab:gray']

    legend = []
    folder_one_path = os.path.join(main_folder, one_name)
    folder_two_path = os.path.join(main_folder, two_name)
    
    one_times = get_exp_times(current_exp=folder_one_path)
    two_times = get_exp_times(current_exp=folder_two_path)
    
    
    all_times = one_times + two_times
    all_times.sort()
    one_times_array = np.array(one_times)
    two_times_array = np.array(two_times)
    x = []
    y_one = []
    y_two = []
    for item in all_times:
        y_one.append((item >= one_times_array).sum())
        y_two.append((item >= two_times_array).sum())
        x.append(item)
    

    plt.plot(x, y_one, colors[index])
    plt.plot(x, y_two, colors[index+1])
    plt.axis((0, max_time, 0, 100))
    legend.append(f'ELS')
    legend.append(f'Our')
    plt.xlabel("Run time")
    plt.ylabel("Solved problems")
    plt.legend(legend, loc='best')
    plt.title(f'One tower - {difficult}')
    plt.savefig(f'image_one_tower_{difficult}_new.png')
    plt.cla()