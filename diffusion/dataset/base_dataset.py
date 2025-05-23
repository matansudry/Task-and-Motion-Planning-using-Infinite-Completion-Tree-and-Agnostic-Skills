import os
from torch.utils.data import Dataset
from abc import abstractmethod

class BaseDataset(Dataset):
    def __init__(self, params:dict):
        """
        init base class

        Args:
            dataset_dirs (list): list of all datasets folders
            preprocesses_cfg (dict): preprocesses config to run
            encode_text (bool, optional): run encode on dataset. Defaults to False.
        """
        self.params = params
        self.device = params.device

    @abstractmethod
    def __len__(self) -> int:
        """
        return number of objects

        Returns:
            int: number of objects
        """

    @abstractmethod
    def __getitem__(self, idx:int) -> dict:
        """
        return sample idx

        Args:
            idx (int): id to return

        Returns:
            dict: sample
        """
        raise
