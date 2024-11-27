import os
from pathlib import Path
import json

import pandas as pd
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

from src.data.datasets import AmazonDataset

class AmazonGraphDataset(AmazonDataset):
    def __init__(self, root, datasetConfig, datasetName, devCtrl):
        super().__init__(root, datasetConfig, datasetName, devCtrl)

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
        
    
    def mapUsersAndItems(self, interactionData, no_nodes_offset_Ctrl=False):
        """Map user IDs and item IDs to unique integers."""
        interactionData['user_id_mapped'] = pd.factorize(interactionData['user_id'])[0]
        user_first_occurrence_df = interactionData.drop_duplicates(subset='user_id_mapped', keep='first')
        user_id_labels, user_id_mapping = pd.factorize(user_first_occurrence_df['user_id'])
        self.user_id_dict = {user: int(label) for user, label in zip(user_first_occurrence_df['user_id'].unique(), user_id_labels)}
        self._saveMappingToFile(self.user_id_dict, file_name=f'user_mapping.json')
        
        no_nodes_offset = len(self.user_id_dict) if no_nodes_offset_Ctrl else 0
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
            
    
    '''def createGraphDataset(self, interactionData):
        """
        Note:
        For training, certain actions are skipped, e.g. filtering out interaction sequences > k.
        """
        edge_index, edge_attr = self.createEdgeIndexAndAttributes(interactionData)
        user_node_features, user_ids, item_node_features, item_ids = self.createNodeFeatures(interactionData)
        data = self.createPyGDataObject(user_node_features, user_ids, item_node_features, item_ids, edge_index, edge_attr)
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
        user_ids = interactionData["user_id_mapped"].to_numpy()
        item_ids = interactionData["parent_asin_mapped"].to_numpy()
        
        user_node_features = torch.Tensor(interactionData["user_id_mapped"].to_numpy()).unsqueeze(dim=-1)
        item_node_features = torch.Tensor(interactionData["parent_asin_mapped"].to_numpy()).unsqueeze(dim=-1)
        
        return user_node_features, user_ids, item_node_features, item_ids


    def createPyGDataObject(self, user_node_features, user_ids, item_node_features, item_ids, edge_index, edge_attr):
        """Create PyTorch Geometric Data object."""
        data = HeteroData()
        
        data['user'].x = user_node_features
        data['user'].node_id = user_ids
        
        data['item'].x = item_node_features
        data['user'].node_id = item_ids
        
        data['user', 'item'].edge_index = edge_index
        data['user', 'item'].y = edge_attr
        
        # We also need to make sure to add the reverse edges from movies to users
        # in order to let a GNN be able to pass messages in both directions.
        data = T.ToUndirected()(data)
        
        return data'''
        
    
    def createGraphDataset(self, interactionData):
        """
        Note:
        For training, certain actions are skipped, e.g. filtering out interaction sequences > k.
        """
        edge_index, edge_attr = self.createEdgeIndexAndAttributes(interactionData)
        user_node_features, user_ids = self.createUserNodeFeatures(interactionData)
        item_node_features, item_ids = self.createItemNodeFeatures(interactionData)
        data = self.createPyGDataObject(user_node_features, user_ids, item_node_features, item_ids, edge_index, edge_attr)
        return data

    def createEdgeIndexAndAttributes(self, interactionData):
        """Create edge_index and edge_attr."""
        user_ids = self.extractUserIDs(interactionData)
        item_ids = self.extractItemIDs(interactionData)
        
        # Create COO edge index
        edge_index_COO_format = torch.tensor([user_ids, item_ids], dtype=torch.long)
        
        # Create edge attributes (e.g., ratings)
        ratings = interactionData["rating"].to_numpy()
        edge_attr = torch.tensor(ratings, dtype=torch.float).view(-1, 1)
        
        return edge_index_COO_format, edge_attr

    def createUserNodeFeatures(self, interactionData):
        """Create user node features and user IDs."""
        user_ids = torch.tensor(pd.Series(self.extractUserIDs(interactionData)).unique(), dtype=torch.long)
        user_node_features = torch.zeros((user_ids.shape[0], 1), dtype=torch.float)
        return user_node_features, user_ids

    def createItemNodeFeatures(self, interactionData):
        """Create item node features and item IDs."""
        item_ids = torch.tensor(pd.Series(self.extractItemIDs(interactionData)).unique(), dtype=torch.long)
        item_node_features = torch.zeros((item_ids.shape[0] + 1, 1), dtype=torch.float)
        return item_node_features, item_ids


    def extractUserIDs(self, interactionData):
        """Extract user IDs from interaction data."""
        return interactionData["user_id_mapped"].to_numpy()

    def extractItemIDs(self, interactionData):
        """Extract item IDs from interaction data."""
        return interactionData["parent_asin_mapped"].to_numpy()

    def createPyGDataObject(self, user_node_features, user_ids, item_node_features, item_ids, edge_index, edge_attr):
        """Create PyTorch Geometric Data object."""
        data = HeteroData()
        
        # Add user data
        data['user'].x = user_node_features
        data['user'].node_id = user_ids
        
        # Add item data
        data['item'].x = item_node_features
        data['item'].node_id = item_ids
        
        # Add edges
        data['user', 'rates', 'item'].edge_index = edge_index
        data['user', 'rates', 'item'].y = edge_attr
        
        # Convert graph to undirected for bidirectional message passing
        data = T.ToUndirected()(data)
        # Convert to homogeneous graph
        # data = data.to_homogeneous()
        
        return data