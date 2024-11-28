from pathlib import Path
import os
import pandas as pd
import numpy as np

from tensorly import decomposition

import torch
from torch.functional import tensordot
from torch import nn, optim, Tensor
import torch_geometric
from torch_geometric.data import Dataset, Data, download_url, extract_zip
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj

from src.data.datasetup import AmazonDatasetSetup


class AmazonGraphDataset(Dataset):
    def __init__(self, datasetSetup, config_dict,
                 transform=None, pre_transform=None,
                 transform_args=None, pre_transform_args=None,
                 rating_threshold=0):
        self.datasetSetup = datasetSetup
        self.config_dict = config_dict
        self.root = Path(self.datasetSetup.root)
        self.transform = transform
        self.pre_transform = pre_transform
        self.transform_args = transform_args
        self.pre_transform_args = pre_transform_args
        self.rating_threshold = rating_threshold
    
    @property
    def raw_file_names(self):
        return self.root / "raw" / "Interactions" / f"{self.datasetSetup.category}_Preprocessed.csv.gz"

    @property
    def processed_file_names(self):
        return f"data_amazon_{self.datasetSetup.category}.pt"
    
    def process(self):
        interactionData = pd.read_csv(self.raw_file_names)
        columns = ["user_id", "parent_asin", "rating"]
        ratings = interactionData.loc[:, columns]
        users = interactionData["user_id"].unique()
        items = interactionData["parent_asin"].unique()
        # TODO: join information about users and items here
        
        # TODO: What is that for?
        num_users = self.config_dict["num_users"]  # self.config_dict
        if num_users != -1:
            users = users[:num_users]
        
        user_ids = range(len(users))
        item_ids = range(len(items))
        
        user_to_id = dict(zip(users, user_ids))
        item_to_id = dict(zip(items, item_ids))
        
        # get adjacency info
        self.num_user = users.shape[0]
        self.num_item = items.shape[0]

        # initialize the adjacency matrix
        rat = torch.zeros(self.num_user, self.num_item)

        for index, row in ratings.iterrows():
            user, item, rating = row[:3]
            if num_users != -1:
                if user not in user_to_id: break
            # create ratings matrix where (i, j) entry represents the ratings
            # of movie j given by user i.
            rat[user_to_id[user], item_to_id[item]] = rating
            
        # create Data object
        data = Data(edge_index = rat,
                    raw_edge_index = rat.clone(),
                    data = ratings,
                    users = users,
                    items = items)
        
        # apply any pre-transformation
        if self.pre_transform is not None:
            data = self.pre_transform(data, self.pre_transform_args)

        # apply any post_transformation
        # if self.transform is not None:
        #     # data = self.transform(data, self.transform_args)
        data = self.transform(data, [self.rating_threshold])
        
        # save important attributes to class
        self.user_to_id = user_to_id
        self.item_to_id = item_to_id
        
        # save the processed data into .pt file
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save(data, os.path.join(self.processed_dir, f"data_amazon_{self.datasetSetup.category}.pt"))
        print('Processing finished.')
    
    def len(self):
        """
        return the number of examples in your graph
        """
        # TODO: how to define number of examples
        return 1
    
    def indices(self):
        """
        Return indices of the dataset. This is typically a sequence.
        Since we have one graph, this will return [0].
        """
        return range(self.len())

    def get(self):
        """
        The logic to load a single graph
        """
        data = torch.load(os.path.join(self.processed_dir, f"data_amazon_{self.datasetSetup.category}.pt"))
        return data
    
    def train_val_test_split(self, val_frac=0.2, test_frac=0.1):
        """
        Return two mask matrices (M, N) that represents edges present in the
        train and validation set
        """
        try:
            self.num_user, self.num_item
        except AttributeError:
            data = self.get()
            self.num_user = len(data["users"].unique())
            self.num_item = len(data["items"].unique())
        # get number of edges masked for training and validation
        num_train_replaced = \
            round((test_frac+val_frac)*self.num_user*self.num_item)
        num_val_show = round(val_frac*self.num_user*self.num_item)

        # edges masked during training
        indices_user = np.random.randint(0, self.num_user, num_train_replaced)
        indices_item = np.random.randint(0, self.num_item, num_train_replaced)
        
        # sample part of edges from training stage to be unmasked during
        # validation
        indices_val_user = np.random.choice(indices_user, num_val_show)
        indices_val_item = np.random.choice(indices_item, num_val_show)

        train_mask = torch.ones(self.num_user, self.num_item)
        train_mask[indices_user, indices_item] = 0

        val_mask = train_mask.clone()
        val_mask[indices_val_user, indices_val_item] = 1

        test_mask = torch.ones_like(train_mask)

        return train_mask, val_mask, test_mask