import sys
sys.path.append(".")

import os
import tqdm
from utils.general_utils import save_pickle, load_pickle

def fix_sample_paths(sample:dict):
    fix = False
    if "no_git/dataset" not in sample['start_state_path']:
        sample['start_state_path'] = sample['start_state_path'].replace(
            "no_git/playground/samples", "no_git/dataset"
        )
        fix=True
    if "no_git/dataset" not in sample['end_state_path']:
        sample['end_state_path'] = sample['end_state_path'].replace(
            "no_git/playground/samples", "no_git/dataset"
        )
        fix=True
    if sample['previous_action'] is not None and "no_git/dataset" not in sample['previous_action']:
        sample['previous_action'] = sample['previous_action'].replace(
            "no_git/playground/samples", "no_git/dataset"
        )
        fix=True
    return sample, fix

if __name__ == "__main__":
    folder_path = "no_git/dataset"
    primitives = os.listdir(folder_path)
    for primitive in primitives:
        temp_path = os.path.join(folder_path,primitive)
        sub_primitives = os.listdir(temp_path)
        for file in tqdm.tqdm(sub_primitives):
            if "P" not in file:
                temp_path_sub = os.path.join(temp_path, file, "sample.pickle")
                sample = load_pickle(temp_path_sub)
                sample, fix = fix_sample_paths(sample=sample)
                if fix:
                    save_pickle(path=temp_path_sub, object_to_save=sample)
            else:
                temp_path_sub = os.path.join(temp_path, file)
                sub_primitives = os.listdir(temp_path_sub)
                for sub_file in tqdm.tqdm(sub_primitives):
                    if "P" not in sub_file:
                        temp_path_sub_sub = os.path.join(temp_path_sub, sub_file, "sample.pickle")
                        sample = load_pickle(temp_path_sub_sub)
                        sample, fix = fix_sample_paths(sample=sample)
                        if fix:
                            save_pickle(path=temp_path_sub_sub, object_to_save=sample)