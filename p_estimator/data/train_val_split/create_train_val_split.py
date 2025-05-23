import sys
sys.path.append(".")

import argparse
import os
from torch.utils.data import random_split
import numpy as np
import random
import tqdm
from utils.general_utils import save_pickle, load_pickle

def main(primitive:str, folderes:str, output_path:str):
    data = {}
    clean_list = {}
    for folder_path in tqdm.tqdm(folderes):
        if not os.path.isdir(folder_path):
            continue
        exps = os.listdir(folder_path)
        for exp in exps:
            exp_path = os.path.join(folder_path, exp)
            for time in os.listdir(exp_path):
                full_exp_path = os.path.join(exp_path, time, "actions")
                if not os.path.isdir(full_exp_path):
                    continue
                actions = os.listdir(full_exp_path)
                for action in actions:
                    if "pickle" not in action or primitive not in action:
                        continue
                    full_sample_path = os.path.join(full_exp_path, action)
                    raw_data = load_pickle(full_sample_path)
                    state = raw_data[0]['observation']
                    high_level_action = raw_data[0]['high_level_action']
                    if high_level_action not in clean_list.keys():
                        clean_list[high_level_action] = np.array(state).reshape(1, -1)
                        data[high_level_action] = [full_sample_path]
                    else:
                        np_state = np.array(state).reshape(1, -1)
                        dist = np.linalg.norm(clean_list[high_level_action] - np_state, axis=1)
                        if np.min(dist) == 0.0:
                            continue
                        clean_list[high_level_action] = np.concatenate([clean_list[high_level_action], np_state], axis=0)
                        data[high_level_action].append(full_sample_path)
    output = {
        "train": [],
        "val": [],
        "test": [],
    }
    for action in data.keys():
        action_data = data[action]
        random.shuffle(action_data)
        train_ratio = int(0.7 * len(action_data))
        val_ratio = int(0.15 * len(action_data))
        test_ratio = len(action_data) - train_ratio - val_ratio
        train = action_data[:train_ratio]
        val = action_data[train_ratio: train_ratio+val_ratio]
        test = action_data[train_ratio+val_ratio:train_ratio+val_ratio+test_ratio]
        output["train"] += train
        output["val"] += val
        output["test"] += test

    print("train:", len(output["train"]))
    print("val:", len(output["val"]))
    print("test:", len(output["test"]))
    #assert not os.path.exists(output_path)
    
    save_pickle(
        path = output_path,
        object_to_save= output
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, default="no_git/system/tamp_dataset_for_training_hard_7200_actions_exp_with_first_action_score")
    parser.add_argument('--primitive', type=str, default="Pick",\
        choices=["Pick", "Place"])
    parser.add_argument('--output_path', type=str, default="p_estimator/data/Pick_split_4.pickle")

    args = parser.parse_args()

    assert args.primitive in args.output_path

    folderes = [f'no_git/system/data_generation/tamp_dataset_generation_round_three_{index}_600' for index in range(120)]
    folderes += [f'no_git/system/data_generation/tamp_dataset_generation_round_two_{index}_600' for index in range(120)]
    
    for folder in folderes:
        assert os.path.isdir(folder)
    main(
        primitive=args.primitive,
        folderes=folderes,
        output_path=args.output_path
    )
    print("done")