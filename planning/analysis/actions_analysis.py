import sys
sys.path.append(".")

import os
import numpy as np
from utils.general_utils import load_pickle
import matplotlib.pyplot as plt

def most_frequent(List):
    _, counts = np.unique(List, return_counts=True)
    return np.max(counts)

def collect_data(data:dict, primitive:str, index:int):
    #collect primtives
    temp_analysis = []
    #single_analysis = []
    for item in data.values():
        if primitive in item:
            temp_analysis.append(item[primitive])
    
    all = []
    success_std = []
    emprical_success_rate = []
    q_values = []

    for item in temp_analysis:
        #all 
        concat_item = np.array(item['all'])
        concat_item = concat_item[:, index]
        all.append(concat_item)
        

        #success_only
        for success_std_index in range(len(item['success_std'])):
            item['success_std'][success_std_index] = item['success_std'][success_std_index] if isinstance(item['success_std'][success_std_index], np.ndarray) else np.zeros(4)
        concat_item = np.array(item['success_std'])
        concat_item = concat_item[:, index]
        success_std.append(concat_item)

        #success_rate
        for success_rate_index in range(len(item['emprical_success_rate'])):
            item['emprical_success_rate'][success_rate_index] = item['emprical_success_rate'][success_rate_index] if isinstance(item['emprical_success_rate'][success_rate_index], np.ndarray) is not None else np.zeros(1)
        concat_item = np.array(item['emprical_success_rate'])
        emprical_success_rate.append(concat_item)
        q_values.append(np.array(item['q_value']))

    all = np.concatenate(all)
    success_std = np.concatenate(success_std)
    emprical_success_rate = np.concatenate(emprical_success_rate)
    q_values = np.concatenate(q_values)
    
    return all, success_std, emprical_success_rate, q_values


def plot_analysis(all:np.ndarray, success_std:np.ndarray, success_rate:np.ndarray,\
    index:int, primitive:str, q_values:np.ndarray):
    legend = []
    bins = 100
    fig, ax = plt.subplots(1, 3)
    all = all.tolist()
    success_std = success_std.tolist()
    success_rate = success_rate.tolist()
    normilze_success_std = [round(item, 3) for item in success_std]
    normilze_all = [round(item, 3) for item in all]
    normilze_success_rate = [round(item, 3) for item in success_rate]
    normilze_q_value = [round(item, 3) for item in q_values]
    
    
    #success std
    data = normilze_success_std
    plt_index = 0
    ax[plt_index].hist(data, bins=bins, color='skyblue', edgecolor='black')
    ax[plt_index].axis((0, 1, 0, most_frequent(data)+2))
    ax[plt_index].set_xlabel("normilze_success_std")
    ax[plt_index].set_ylabel("count")
    ax[plt_index].legend(legend, loc='best')
    #ax[plt_index].set_title(f'{str(index)}_{primitive}_#samples_{len(data)}')

    #success_rate
    data = normilze_success_rate
    plt_index = 2
    #ax[plt_index].hist(data, bins=bins, color='skyblue', edgecolor='black')
    #ax[plt_index].axis((0, 1, 0, most_frequent(data)+2))
    ax[plt_index].set_xlabel("success_rate - q_value")
    #ax[plt_index].legend(legend, loc='best')
    #ax[plt_index].set_title(f'{str(index)}_{primitive}_#samples_{len(data)}')
    x = np.linspace(0, len(normilze_q_value)-1, len(normilze_q_value))
    y = np.array(normilze_success_rate) - np.array(normilze_q_value)
    ax[plt_index].axis(ymin=-1,ymax=1)
    ax[plt_index].hlines(0, 0, len(normilze_q_value))
    ax[plt_index].plot(x, y, 'o', color='black')


    #all
    data = normilze_all
    plt_index = 1
    ax[plt_index].hist(data, bins=bins, color='skyblue', edgecolor='black')
    ax[plt_index].axis((0, 1, 0, int(most_frequent(data)+2)))
    ax[plt_index].set_xlabel("all")
    ax[plt_index].legend(legend, loc='best')
    #ax[plt_index].set_title(f'{str(index)}_{primitive}_#samples_{len(data)}')
    fig.suptitle(f'{str(index)}_{primitive}_#samples_{len(data)}')

    fig.savefig(f'plots/{str(index)}_{primitive}_plot.png')
    plt.cla()
    
    



if __name__ == "__main__":
    folder_path = "no_git/system/tamp_action_analysis_with_first_action_score_mid_450_actions_exp_with_first_action_score"
    experiments = os.listdir(folder_path)
    output = {}
    primitives = []
    for experiment in experiments:
        output[experiment] = {}
        experiment_path = os.path.join(folder_path, experiment)
        experiment_path = os.path.join(experiment_path, os.listdir(experiment_path)[0], "actions")
        assert os.path.exists(experiment_path)
        for action in os.listdir(experiment_path):
            if "pickle" not in action:
                continue
            exp_actions = load_pickle(os.path.join(experiment_path, action))
            success_action_list = []
            for item in exp_actions:
                if item['reward'] > 0:
                    success_action_list.append(item["low_level_action"])
            success_std = np.std(np.array(success_action_list), axis=0)
            all_action_list = [item["low_level_action"] for item in exp_actions]
            all_std = np.std(np.array(all_action_list), axis=0)
            reward_list = [item["reward"] for item in exp_actions]
            success = sum(reward_list)
            emprical_success_rate = success / len(reward_list)
            q_value = exp_actions[0]['first_action_score']

            
            primitive = action.split("_")[0]
            if primitive not in output[experiment]:
                output[experiment][primitive] = {
                    "all": [],
                    "success_std": [],
                    "emprical_success_rate": [],
                    "q_value": [],
                }
            if primitive not in primitives:
                primitives.append(primitive)
            output[experiment][primitive]["all"].append(all_std)
            output[experiment][primitive]["success_std"].append(success_std)
            output[experiment][primitive]["emprical_success_rate"].append(emprical_success_rate)
            output[experiment][primitive]["q_value"].append(q_value)
        
    for index in range(4):
        for primitive in primitives:
            all, success_std, success_rate, q_values = collect_data(
                data=output,
                primitive=primitive,
                index=index
            )
            plot_analysis(
                all=all,
                success_std=success_std,
                success_rate=success_rate,
                index=index,
                primitive=primitive,
                q_values=q_values,
            )
            
    
