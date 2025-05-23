from torch.utils.data import Dataset
import numpy as np
import tqdm

from utils.general_utils import load_pickle

class PEstimatorDataset(Dataset):
    def __init__(self, dataset_cfg:dict):
        self.cfg = dataset_cfg
        self.data = dataset_cfg['split']
        #self._remove_duplications()

    def _remove_duplications(self):
        indexes = []
        clean_list = {}
        for index in tqdm.tqdm(range(len(self.data))):
            path = self.data[index]
            raw_data = load_pickle(path)
            state = raw_data[0]['observation']
            action = raw_data[0]['high_level_action']
            if action not in clean_list.keys():
                clean_list[action] = np.array(state).reshape(1, -1)
                indexes.append(index)
            else:
                np_state = np.array(state).reshape(1, -1)
                dist = np.linalg.norm(clean_list[action] - np_state, axis=1)
                if np.min(dist) == 0.0:
                    raise
                clean_list[action] = np.concatenate([clean_list[action], np_state], axis=0)
                indexes.append(index)
        
        #update data
        self.data = [self.data[i] for i in indexes]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx:int):
        path = self.data[idx]
        raw_data = load_pickle(path)
        
        sample = {}
        sample["state"] = raw_data[0]['observation']
        sample["high_level_action"] = raw_data[0]['high_level_action']

        reward_list = [item["reward"] for item in raw_data]
        success = sum(reward_list)
        sample["success_rate"] = (success / len(reward_list))
        sample['first_action_score'] = raw_data[0]['first_action_score']


        return sample