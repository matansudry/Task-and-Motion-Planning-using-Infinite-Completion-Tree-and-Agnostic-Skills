import sys
sys.path.append(".")

import os
from utils.general_utils import load_pickle
from utils.general_utils import load_pickle
import matplotlib.pyplot as plt

def get_success_rate(low_level_graph, high_level_graph):
    #load loggers
    final_dict = {}
    all_actions = []
    
    for node_key in low_level_graph.graph.nodes._nodes:
        node = low_level_graph.graph.nodes._nodes[node_key]
        final_dict[node_key] = {}
        node_mapping = {}
        for edge_key in low_level_graph.graph.edges._adjdict[node_key].keys():
            temp_high_level_key =\
                low_level_graph.graph.edges._adjdict[node_key][edge_key]['high_level_action_id']
            if temp_high_level_key not in node_mapping:
                node_mapping[temp_high_level_key] = 1
            else:
                node_mapping[temp_high_level_key] += 1
        for _, edge_try in enumerate(node['tries']):
            num_of_tries = node['tries'][edge_try]
            num_of_success = node_mapping[edge_try] if edge_try in node_mapping else 0
            
            
            action = high_level_graph.graph.nodes._nodes[edge_try]['action'].name
            high_level_action = action.split("(")[1].split(" ")[0]
            high_level_action_from = action.split("(")[1].split(" ")[2].split(")")[0]
            if high_level_action_from != "table" and high_level_action_from != "rack":
                high_level_action_from = "box"
            full_action = high_level_action +"_"+ high_level_action_from
            if full_action not in all_actions:
                all_actions.append(full_action)

            final_dict[node_key][edge_try] = {
                "num_of_tries": num_of_tries,
                "num_of_success": num_of_success,
                "success_rate": num_of_success/num_of_tries,
                "type": full_action
            }
    
    return final_dict, all_actions


def get_how_much_high_level_plans(high_level_graph):
    cnt = 0
    for item in high_level_graph.graph.nodes._nodes.values():
        cnt += 1 if item['goal_reached'] else 0
    
    return cnt
    

def plot_success_rate_as_tries(success_rate:dict, name:str, max_x:int):
    cnt = 0
    x = []
    y = []
    for item in success_rate.values():
        for item_2 in item.values():
            x.append(cnt)
            y.append(item_2['success_rate'])
            cnt += 1

    plt.plot(x, y, 'bx')
    plt.axis((0, max_x, 0, 1))
    legend = []
    legend.append(f'{name}')
    plt.xlabel("tries")
    plt.ylabel("success_rate")
    plt.legend(legend, loc='best')
    plt.title(f'success_rate_as_tries')
    plt.savefig(f'success_rate_as_tries_{name}.png')
    plt.cla()
    
def plot_success_rate_as_tries_hist(success_rate:dict, name:str,  action:str):
    data = []
    for item in success_rate.values():
        for item_2 in item.values():
            if item_2['type'] == action:
                data.append(round(item_2['success_rate'],2))

    plt.hist(data, bins=100, color='skyblue', edgecolor='black')
    plt.axis((0, 1, 0, 200))
    legend = []
    legend.append(f'{name}')
    plt.xlabel("success_rate")
    plt.ylabel("count")
    plt.legend(legend, loc='best')
    plt.title(f'action_{action}_count_as_success_rate')
    plt.savefig(f'plots/problem_analysis/action_{action}_count_as_success_rate_hist_{name}.png')
    plt.cla()
    

if __name__ == "__main__":
    main_folder = "no_git/system/"
    names = {
        "els": "els_v2_one_tower_mid_1200",
        "tamp": "tamp_one_tower_mid_1200"
    }
    for output_name in ["els", "tamp"]:
        name = names[output_name]
        assert output_name in name
        problem = "77_single"
        max_time = 1200
        
        #load ELS paths
        problem_path = os.path.join(main_folder, name, problem)
        problem_path = os.path.join(problem_path, list(os.listdir(problem_path))[0])
        log_path = os.path.join(problem_path, "log.pickle")
        logger = load_pickle(log_path)
        low_level_graph_path = os.path.join(problem_path, "low_level_grpah.gpickle")
        low_level_graph = load_pickle(low_level_graph_path)
        high_level_graph_path = os.path.join(problem_path, "high_level_grpah.gpickle")
        high_level_graph = load_pickle(high_level_graph_path)
        
        #high_level_plans = get_how_much_high_level_plans(high_level_graph=high_level_graph)

        
        success_rate, all_actions = get_success_rate(
            low_level_graph=low_level_graph,
            high_level_graph=high_level_graph
        )
        for action in all_actions:
            plot_success_rate_as_tries_hist(
                success_rate=success_rate,
                name=output_name,
                action=action
            )
