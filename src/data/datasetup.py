import os
from pathlib import Path
from urllib.parse import urljoin
import pprint
import gzip
import json

import pandas as pd

from src.data.utils import loadFileFromURL
from src.utils.wrapper import tryExcept, timeMeasured


class AmazonDatasetSetup:
    """
    Dataset class for the Amazon Review Dataset from 2023.
    Overview: https://amazon-reviews-2023.github.io/main.html
    """
    
    def __init__(self, root, datasetConfig, datasetName, devCtrl=False):
        self.devCtrl = devCtrl
        self.datasetConfig = datasetConfig
        self.datasetName = datasetName
        
        print(f"Reading config file at {datasetConfig}")
        with open(datasetConfig, "r") as configFile:
            configData = json.load(configFile)
            self.datasetConfig = configData.get(datasetName, {})
            print(f"Dataset Config set as:")
            pprint.pp(self.datasetConfig)
            
        self.category = self.datasetConfig.get("category")
        self.root = Path(root) / f"{self.category}"
        
        urls = self.datasetConfig.get("urls", {})
        self.interactionDataUrl = urls.get("interactionDataUrl", "default_interaction_data_url")
        self.metaDataUrl = urls.get("metaDataUrl", "default_meta_data_url")
        self.reviewDataUrl = urls.get("reviewDataUrl", "default_review_data_url")
        
        filterCriteria = self.datasetConfig.get("filterCriteria", {})
        self.requiredFields = filterCriteria.get("requiredFields")
        self.filterItemsBasedOnFeatures = filterCriteria.get("filterItemsBasedOnFeatures")
        self.filterkInteractions = filterCriteria.get("filterkInteractions")
        
        self.rawDataDir = self.root / "raw"
        self.rawDataDir.mkdir(parents=True, exist_ok=True)
        self.rawUnwrappedItemDataDir = self.rawDataDir / "Items" / "Unwrapped"
        self.rawUnwrappedItemDataDir.mkdir(parents=True, exist_ok=True)

        if not os.path.exists(self.root / "raw" / "Interactions" / f"{self.category}_Preprocessed.csv.gz"): 
            self.downloadInteractionData()
            self.downloadItemDataAsJSON()
            self.unwrapItemData(self.rawMetaDatasetPath)
            self.preprocessInteractionData()
            self.deleteUnusedItems()
        else:
            print(f"Dataset already downloaded and preprocessed, no further action from DatasetSetup object.")
        
        self.getUsersItemsCount()
    
    
    @tryExcept
    @timeMeasured
    def downloadInteractionData(self):
        datasetFilename = f"{self.category}.csv.gz"
        datasetPath = self.rawDataDir / "Interactions" / datasetFilename
        datasetPath.parent.mkdir(parents=True, exist_ok=True)
        if not datasetPath.exists():
            datasetUrl = urljoin(self.interactionDataUrl, datasetFilename)
            loadFileFromURL(datasetUrl, datasetPath)
    
    @tryExcept
    @timeMeasured
    def downloadItemDataAsJSON(self):
        metaDataFilename = f"meta_{self.category}.jsonl.gz"
        self.rawMetaDatasetPath = self.rawDataDir / "Items" / metaDataFilename
        self.rawMetaDatasetPath.parent.mkdir(parents=True, exist_ok=True)
        if not self.rawMetaDatasetPath.exists():
            loadFileFromURL(urljoin(self.metaDataUrl, metaDataFilename), self.rawMetaDatasetPath)
        
        reviewDataFilename = f"{self.category}.jsonl.gz"
        self.rawReviewDatasetPath = self.rawDataDir / "Items" / reviewDataFilename
        self.rawReviewDatasetPath.parent.mkdir(parents=True, exist_ok=True)
        if not self.rawReviewDatasetPath.exists():
            loadFileFromURL(urljoin(self.reviewDataUrl, reviewDataFilename), self.rawReviewDatasetPath)
    
    
    @tryExcept
    @timeMeasured
    def unwrapItemData(self, datasetPath):        
        with gzip.open(datasetPath, "rt", encoding="utf-8") as f:
            linesCount, self.dumpedJSONlist = 0, []
            for line in f:
                linesCount += 1
                jsonLine = json.loads(line.strip())
                if self._checkRequiredFields(jsonLine):
                    parent_asin = jsonLine.get("parent_asin")
                    outputFilePath = self.rawUnwrappedItemDataDir / f"{parent_asin}.json"
                    with open(outputFilePath, "w", encoding="utf-8") as outputFile:
                        json.dump(jsonLine, outputFile, indent=4)
                    self.dumpedJSONlist.append(parent_asin)
                if linesCount == 100 and self.devCtrl:
                    print(f"Dev control on, loading dataset stopped at {linesCount} lines.")
                    break
        
        print(f"Unwrapped dataset from {datasetPath}")
        print(f"Saved {len(self.dumpedJSONlist)} from a total of {linesCount} lines.")
    
    
    def _checkRequiredFields(self, jsonLine):
        """
        Checks whether the required fields are present based on the config.
        Returns True if all required fields are valid; False otherwise.
        """
        for field, isRequired in self.requiredFields.items():
            if isRequired:
                value = jsonLine.get(field)
                # For description and images, we ensure they are non-empty lists
                if field == "description" or field == "images":
                    if not isinstance(value, list) or not value:
                        return False
                # For other fields, just check if they are truthy (non-null, non-empty)
                elif not value:
                    return False
        return True
    
    
    @tryExcept
    @timeMeasured
    def preprocessInteractionData(self):
        """
        Filters and saves interaction data.
        """
        interactionDataRawPath = self.root / "raw" / "Interactions" / f"{self.category}.csv.gz"
        interactionData = pd.read_csv(interactionDataRawPath)
        
        if self.filterItemsBasedOnFeatures:
            interactionData = self._filterInteractionItemsBasedOnFeatures(interactionData)
        
        if int(self.filterkInteractions) > 0:
            interactionData = self._filterFrequentUsersandItemsIter(interactionData, k=self.filterkInteractions)
        
        self.interactionData = interactionData
        interactionDataPreprocessedPath = self.root / "raw" / "Interactions" / f"{self.category}_Preprocessed.csv.gz"
        interactionData.to_csv(interactionDataPreprocessedPath, index=False, compression='gzip')
    
    def _filterInteractionItemsBasedOnFeatures(self, interactionData):
        itemFilteredBasedOnFeaturesListMask = [s.replace(".JSON", "").replace(".json", "") for s in os.listdir(self.rawUnwrappedItemDataDir)]
        return interactionData[interactionData["parent_asin"].isin(itemFilteredBasedOnFeaturesListMask)]
        
          
    '''
    def _filterFrequentUsersandItems(self, interactionData, drop="user", k=10):
        """
        Filter nodes with at least k interactions.
        Method without iterations, can only drop either(!) user or item, controlled with param 'drop'.
        """
        if drop == "user":
            frequent_users = interactionData["user_id"].value_counts()
            frequent_users_mask = frequent_users[frequent_users >= k].index.tolist()
            interactionData = interactionData[interactionData["user_id"].isin(frequent_users_mask)]
        
        elif drop == "item":
            frequent_items = interactionData["parent_asin"].value_counts()
            frequent_items_mask = frequent_items[frequent_items >= k].index.tolist()
            interactionData = interactionData[interactionData["parent_asin"].isin(frequent_items_mask)]
        
        return interactionData
    '''
    
    
    def _filterFrequentUsersandItemsIter(self, interactionData, k=10):
        """
        Filter users and items with at least k interactions each.
        This ensures that both users and items meet the frequency threshold.
        """
        while True:
            user_counts = interactionData["user_id"].value_counts()
            frequent_users_mask = user_counts[user_counts >= k].index.tolist()

            item_counts = interactionData["parent_asin"].value_counts()
            frequent_items_mask = item_counts[item_counts >= k].index.tolist()

            filtered_data = interactionData[
                interactionData["user_id"].isin(frequent_users_mask) &
                interactionData["parent_asin"].isin(frequent_items_mask)
            ]
            # Break if no further filtering is needed (i.e., convergence)
            if len(filtered_data) == len(interactionData):
                break
            
            interactionData = filtered_data

        return filtered_data
    
    
    def deleteUnusedItems(self, columnName="parent_asin"):
        """
        Deletes files or directories under a given path that are not listed in the specified column.

        Parameters:
        - interactionData (pd.DataFrame): The DataFrame containing the valid item identifiers.
        - column (str): The column in the DataFrame that contains the valid item identifiers (e.g., "parent_asin").

        Returns:
        - None
        """
        print("Deleting not needed item jsons now.")
        print(f"File number before cleaning: {len(os.listdir(self.rawUnwrappedItemDataDir))}")
        valid_items = set(self.interactionData[columnName])
        
        for item_filename in os.listdir(self.rawUnwrappedItemDataDir):
            item_path = os.path.join(self.rawUnwrappedItemDataDir, item_filename)

            item_name = item_filename.replace(".json", "")
            if item_name not in valid_items:
                if os.path.isfile(item_path):
                    os.remove(item_path)
        
        print(f"File number after cleaning: {len(os.listdir(self.rawUnwrappedItemDataDir))}")
        
        
    def getUsersItemsCount(self):
        self.interactionData = pd.read_csv(self.root / "raw" / "Interactions" / f"{self.category}_Preprocessed.csv.gz")
        no_users = len(self.interactionData["user_id"].unique())
        no_items = len(self.interactionData["parent_asin"].unique())
        print(f"Number of users in dataset: {no_users, no_items}")