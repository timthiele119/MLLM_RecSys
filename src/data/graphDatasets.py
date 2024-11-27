import os
import json

import torch
from datasets import AmazonDataset
from torch_geometric.data import Data


class AmazonGraphDataset(AmazonDataset):
    def __init__(self, root, datasetConfig, datasetName, transform=None, pre_transform=None):
        super().__init__(root, datasetConfig, datasetName)
        self.transform = transform
        self.pre_transform = pre_transform
        self.processedItemMetaDataDir = self.rawDataDir / "Items" / "Processed"
        if not os.path.exists(self.processedItemMetaDataDir):
            self.process()

    @property
    def raw_file_names(self):
        return list(self.rawUnwrappedItemDataDir)
    
    @property
    def processed_file_names(self):
        """If available, process function does not get triggered. Else, processes raw data."""
        return list(self.processedItemMetaDataDir)
    
    def download(self):
        pass
    
    def process(self):
        """
        Nodes: Items, Users
        Edges: Interactions
        Node Features: Item Features, Users don't have any
        Labels: Item or User?
        """
        
        # filter out data
        
        # 1) only use (meta) data inside certain timeframe -> NOT RIGHT NOW
        
        # 2) only use data with interaction count > threshold = 10
        #     first users, drop, then items

        # 3) negative sampling strategy
        
        data_list = []
        for file_path in self.rawDataDir.glob("*.json"):
            with open(file_path, "r") as f:
                json_line = json.load(f)
                # Process the data into a graph, here we are assuming nodes and edges
                # Example: This could be the node features and edge connections based on user-item interactions
                x = torch.tensor([json_line.get('feature', [0.0])], dtype=torch.float)  # Example node features
                edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)  # Example edge index (user-item graph)
                data = Data(x=x, 
                            edge_index=edge_index,
                            edge_attr=edge_weights,
                            y=label
                            )

                if self.pre_transform:
                    data = self.pre_transform(data)

                data_list.append(data)

        torch.save(self.collate(data_list), os.path.join(self.processed_dir, 'data.pt'))
        
    # construct users data
    # construct item data
    # construct edges

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))



if __name__ == "__main__":
    
    root="data/AmazonReviews"
    os.makedirs(root, exist_ok=True)
    datasetConfigAmazon = "src/data/datasetConfigAmazon.json"
    datasetName = "AmazonAllBeautyDataset"
    AmazonBeautyDataset = AmazonGraphDataset(root, datasetConfigAmazon, datasetName)