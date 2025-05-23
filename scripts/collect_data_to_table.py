import sys
sys.path.append(".")

import os
import tqdm
from utils.general_utils import save_pickle, load_pickle
import pandas as pd

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

if __name__ == "__main__":
    folder_path = "no_git/dataset"
    output = {
        'start_observation': [],
        'end_observation': [],
        'primitive': [],
        'reward': [],
        'action_0': [],
        'action_1': [],
        'action_2': [],
        'action_3': [],
        'timestamp': [],
        'previous_action': [],
        'start_state_path': [],
        'end_state_path': [],
        "previous_primitive": [],
        "sample_path": [],
    }
    primitives = os.listdir(folder_path)
    for primitive in primitives:
        primitive_folder_path = os.path.join(folder_path, primitive)
        samples_names = os.listdir(primitive_folder_path)
        for sample_name in samples_names:
            if "P" not in sample_name:
                full_sample_path = os.path.join(
                    primitive_folder_path, sample_name, "sample.pickle"
                    )
                sample = load_pickle(path=full_sample_path)
                output = update_dict(sample=sample, table=output, sample_path=full_sample_path)
            else:
                primitive_folder_path_2 = os.path.join(primitive_folder_path, sample_name)
                samples_names_2 = os.listdir(primitive_folder_path_2)
                for sample_name_2 in samples_names_2:
                    if "P" not in sample_name_2:
                        full_sample_path = os.path.join(
                            primitive_folder_path_2, sample_name_2, "sample.pickle"
                            )
                        sample = load_pickle(path=full_sample_path)
                        output = update_dict(sample=sample, table=output, sample_path=full_sample_path)
    df = pd.DataFrame.from_dict(output)
    df.to_parquet('no_git/data_summary_17_2_2024.gzip',
                compression='gzip')  
    #pd.read_parquet('df.parquet.gzip')
    temp=pd.read_parquet('no_git/data_summary_17_2_2024.gzip')