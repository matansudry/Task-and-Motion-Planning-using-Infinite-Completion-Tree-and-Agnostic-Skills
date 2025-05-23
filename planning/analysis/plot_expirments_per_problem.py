import sys
sys.path.append(".")

import argparse
import os
import pandas as pd
import numpy as np
from utils.general_utils import load_pickle
from planning.analyser import logger_to_df

def get_exp_times(current_exp:str):
    #iterate on folder one
    runs = list(os.listdir(current_exp))
    current_exp_times = {}
    for run in runs:
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
                current_exp_times[run] = logger["total_time"]
    
    return current_exp_times

if __name__ == "__main__":
    main_folder = "no_git/system/"
    one_name = "els_v2_one_tower_mid_1200"
    two_name = "tamp_one_tower_mid_1200"
    max_time = 1200
    
    folder_one_path = os.path.join(main_folder, one_name)
    folder_two_path = os.path.join(main_folder, two_name)
    
    one_times = get_exp_times(current_exp=folder_one_path)
    two_times = get_exp_times(current_exp=folder_two_path)
    

    x_els = []
    x_our = []
    x_both = []
    x_none = []
    diff_els = []
    diff_our = []
    diff_both = []
    diff_none = []
    for item in one_times:
        if item in two_times:
            if one_times[item] - two_times[item] < 0:
                temp=1
            diff_both.append(one_times[item] - two_times[item])
            x_both.append(int(item.split("_")[0]))
        else:
            diff_els.append(-max_time)
            x_els.append(int(item.split("_")[0]))
    for item in two_times:
        if item not in one_times:
            diff_our.append(max_time)
            x_our.append(int(item.split("_")[0]))
    
    import matplotlib.pyplot as plt

    #plt.plot(x, y_one, 'ro')
    #plt.plot(x, y_two, 'bo')
    plt.plot(x_both, diff_both,'bx', markeredgewidth=2)
    plt.plot(x_els, diff_els,'gx', markeredgewidth=2)
    plt.plot(x_our, diff_our,'rx', markeredgewidth=2)
    x1=np.array([0,100])
    y1=np.array([0,0])
    plt.plot(x1,y1)
    #plt.axis((0, 100, 0, 60))
    plt.title("Diff ELS - OUR")
    legend = []
    #legend.append("ELS")
    #legend.append("Our")
    legend.append("both solved")
    legend.append("ELS solved")
    legend.append("Our solved")
    plt.xlabel("Run time")
    plt.ylabel("Solved problems")
    #plt.fill_between(x, y_one, y_two, color="grey", alpha=0.3)
    plt.legend(legend, loc='best')
    plt.savefig("image_per_problem_mid_1200.png")
    plt.cla()
    temp=1