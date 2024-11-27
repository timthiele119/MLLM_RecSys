import os
from pathlib import Path
import json

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from src.utils.wrapper import tryExcept, timeMeasured
from src.data.datasets import AmazonDataset

class AmazonGraphDataset(AmazonDataset):
    def __init__(self, root, datasetConfig, datasetName):
        super().__init__(root, datasetConfig, datasetName)

        trainingSetInteractionData = self.loadInteractionData("train", filter_users=True, k=10)
        validationSetInteractionData = self.loadInteractionData("valid")
        testSetInteractionData = self.loadInteractionData("test")
        interactionData = pd.concat([trainingSetInteractionData, validationSetInteractionData, testSetInteractionData])
        self.interactionData = interactionData
        del trainingSetInteractionData
        del validationSetInteractionData
        del testSetInteractionData
        
        self.mappingsDataDir = self.root / "Mappings"
        self.mappingsDataDir.mkdir(parents=True, exist_ok=True)
        self.mapUsersAndItems(interactionData)
        
        self.trainingData = self.createGraphDataset(interactionData[interactionData["Split"] == "train"])
        self.validationData = self.createGraphDataset(interactionData[interactionData["Split"] == "valid"])
        self.testData = self.createGraphDataset(interactionData[interactionData["Split"] == "test"])

    
    def loadInteractionData(self, split: str, filter_users: bool = False, k: int = 10):
            interactionDataPath = self.root / "raw" / "Interactions" / f"{self.category}.{split}.csv.gz"
            interactionData = pd.read_csv(interactionDataPath)
            if filter_users and split == "train":
                interactionData = self._filterFrequentUsers(interactionData, k=k)
            interactionData["Split"] = split
            return interactionData
        
        
    def _filterFrequentUsers(self, interactionData, k=10):
        """Filter users with at least k interactions."""
        frequent_users = interactionData["user_id"].value_counts()
        frequent_users_mask = frequent_users[frequent_users > k].index.tolist()
        return interactionData[interactionData["user_id"].isin(frequent_users_mask)]
        
    
    def mapUsersAndItems(self, interactionData):
        """Map user IDs and item IDs to unique integers."""
        interactionData['user_id_mapped'] = pd.factorize(interactionData['user_id'])[0]
        user_first_occurrence_df = interactionData.drop_duplicates(subset='user_id_mapped', keep='first')
        user_id_labels, user_id_mapping = pd.factorize(user_first_occurrence_df['user_id'])
        self.user_id_dict = {user: int(label) for user, label in zip(user_first_occurrence_df['user_id'].unique(), user_id_labels)}
        self._saveMappingToFile(self.user_id_dict, file_name=f'user_mapping.json')
        
        no_nodes_offset = len(self.user_id_dict)
        interactionData['parent_asin_mapped'] = pd.factorize(interactionData['parent_asin'])[0] + no_nodes_offset
        item_first_occurrence_df = interactionData.drop_duplicates(subset='parent_asin_mapped', keep='first')
        item_id_labels, item_id_mapping = pd.factorize(item_first_occurrence_df['parent_asin'])
        self.item_id_dict = {item: int(label) + no_nodes_offset for item, label in zip(item_first_occurrence_df['parent_asin'].unique(), item_id_labels)}
        self._saveMappingToFile(self.item_id_dict, file_name=f'item_mapping.json')
    
    
    def _saveMappingToFile(self, mapping_dict, file_name):
        """Helper function to save a dictionary to a JSON file."""
        mapping_file_path = self.mappingsDataDir / file_name
        with open(mapping_file_path, 'w') as json_file:
            json.dump(mapping_dict, json_file, indent=4)
            print(f"Saved {file_name} to {self.mappingsDataDir}")
            
    
    def createGraphDataset(self, interactionData):
        """
        Note:
        For training, certain actions are skipped, e.g. filtering out interaction sequences > k.
        """
        edge_index, edge_attr = self.createEdgeIndexAndAttributes(interactionData)
        user_node_features, item_node_features = self.createNodeFeatures(interactionData)
        data = self.createPyGDataObject(user_node_features, item_node_features, edge_index, edge_attr)
        return data
    
    
    def createEdgeIndexAndAttributes(self, interactionData):
        """Create edge_index and edge_attr."""
        user_ids = interactionData["user_id_mapped"].to_numpy()
        item_ids = interactionData["parent_asin_mapped"].to_numpy()

        edge_index_COO_format = torch.tensor([user_ids, item_ids], dtype=torch.long)
        
        ratings = interactionData["rating"].to_numpy()
        edge_attr = torch.tensor(ratings, dtype=torch.float).view(-1, 1)

        return edge_index_COO_format, edge_attr
    
    
    def createNodeFeatures(self, interactionData):
        """
        Create node features for users and items.
        TODO: Right now, no real features, just indexing. Extract features for both users and items.
        """
        user_node_features = torch.Tensor(interactionData["user_id_mapped"].to_numpy()).unsqueeze(dim=-1)
        item_node_features = torch.Tensor(interactionData["parent_asin_mapped"].to_numpy()).unsqueeze(dim=-1)
        return user_node_features, item_node_features


    def createPyGDataObject(self, user_node_features, item_node_features, edge_index, edge_attr):
        """Create PyTorch Geometric Data object."""
        data = HeteroData()
        data['user'].x = user_node_features
        data['item'].x = item_node_features
        data['user', 'item'].edge_index = edge_index
        data['user', 'item'].y = edge_attr
        return data