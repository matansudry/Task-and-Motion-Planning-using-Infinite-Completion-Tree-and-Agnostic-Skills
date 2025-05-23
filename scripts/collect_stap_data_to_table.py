import sys
sys.path.append(".")

import os
import tqdm
from utils.general_utils import save_pickle, load_pickle
import pandas as pd
import numpy as np


def update_dict(sample:dict, table:dict, sample_path:str):
    table['start_observation'].append(sample['start_observation'])
    table['end_observation'].append(sample['end_observation'])
    table['primitive'].append(sample['primitive'])
    table['reward'].append(sample['action'][0])
    table['action_0'].append(sample['action'][0])
    table['action_1'].append(sample['action'][1])
    table['action_2'].append(sample['action'][2])
    table['action_3'].append(sample['action'][3])
    table['timestamp'].append(sample['timestamp'])
    table['start_state_path'].append(sample['start_state_path'])
    table['previous_action'].append(sample['previous_action'])
    table['end_state_path'].append(sample['end_state_path'])
    
    if sample['previous_action'] is not None:
        previous_primitive = sample['start_state_path'].split("/")[-4]
        table['previous_primitive'].append(previous_primitive)
    else:
        table['previous_primitive'].append(None)
    table['sample_path'] = sample_path
        
    return table

def get_premitive(path:str):
    options = ["pick", "push", "place", "pull"]
    for option in options:
        if option in path:
            return option
    raise
    

def load_all_folder_files(output:dict, folder_path:str, dataset_type:str, premitive:str):
    all_files = os.listdir(folder_path)
    paths = [os.path.join(folder_path, file) for file in all_files]
    for path in paths:
        batch = dict(np.load(path, allow_pickle=True))
        for index in range(batch['observation'].shape[0]):
            action = batch['action'][index]
            if np.isnan(action[0]):
                output['observation'].append(batch['observation'][index].tolist())
            else:    
                output['next_observation'].append(batch['observation'][index].tolist())
                output['action'].append(action)
                output['reward'].append(batch['reward'][index])
                output['discount'].append(batch['discount'][index])
                output['terminated'].append(batch['terminated'][index])
                output['truncated'].append(batch['truncated'][index])
                output['policy_args'].append(batch['policy_args'][index])
                output['dataset_type'].append(dataset_type)
                output['premitive'].append(premitive)
                output["path"].append(path)
                
                valid = "invalid" if "invalid" in path else "valid"
                output["valid"].append(valid)
            
    return output

if __name__ == "__main__":
    output_path = 'no_git/stap_data_summary_29_2_2024.gzip'
    if os.path.isfile(output_path):
        print("path exist = ", output_path)
        raise
    output = {
        'observation': [],
        'next_observation': [],
        'action': [],
        'reward': [],
        'reward': [],
        'discount': [],
        'terminated': [],
        'truncated': [],
        'policy_args': [],
        "dataset_type": [],
        "premitive": [],
        "path": [],
        "valid": [],
    }
    main_folder_path = "no_git/stap_datasets"
    folders = os.listdir(main_folder_path)
    for folder in tqdm.tqdm(folders):
        temp_folder_path = os.path.join(main_folder_path, folder)
        if not os.path.isdir(temp_folder_path):
            continue
        dataset_type = "validation" if "validation" in temp_folder_path else "train"
        temp_folder_path = os.path.join(temp_folder_path, "train_data")
        premitive = get_premitive(path=temp_folder_path)
        output = load_all_folder_files(
            output=output,
            folder_path=temp_folder_path,
            dataset_type=dataset_type,
            premitive=premitive
        )
    df = pd.DataFrame.from_dict(output)
    df.to_parquet(output_path,
                compression='gzip')  
    #pd.read_parquet('df.parquet.gzip')
    temp=pd.read_parquet(output_path)