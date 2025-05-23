import pandas as pd
import numpy as np
import torch

from diffusion.dataset.base_dataset import BaseDataset


class DataFrameDataset(BaseDataset):
    def __init__(self, params:dict, policy):
        super().__init__(params)
        self._load_dataframes()
        self._filter_dataframes()
        self.policy = policy

    def _load_dataframes(self):
        assert len(self.params['paths']) == 1
        self.df = pd.read_parquet(self.params['paths'][0])
        
    def _filter_dataframes(self):
        #use only primitive == self.params.primitive and previous_primitive is not None
        self.df = self.df.query(f'premitive == "{self.params.primitive}"')
        self.df = self.df.query(f'dataset_type == "{self.params.data_type}"')
        self.df = self.df.query(f'valid == "valid"')
        self.df = self.df.reset_index()

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dict_row = dict(row)
        start_observation = np.stack(dict_row['observation'])
        end_observation = np.stack(dict_row['next_observation'])
        action = dict_row['action']
        obs_indices = dict_row['policy_args']['observation_indices']
        policy_args = dict_row["policy_args"]
        
        sample = {
            "start_observation":start_observation,
            "action":action,
            "end_observation":end_observation,
            "obs_indices":obs_indices,
            "policy_args": policy_args,
            }
        
        return sample
