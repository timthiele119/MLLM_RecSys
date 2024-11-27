import os
from pathlib import Path
from urllib.parse import urljoin
import requests
import pprint
import gzip
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
# from torch_geometric.data import Dataset, download_url

from datasets import load_dataset

from src.data.utils import loadFileFromURL
from src.utils.wrapper import tryExcept, timeMeasured


class AmazonDataset(Dataset):
    """
    Dataset class for the Amazon Review Dataset from 2023.
    Overview: https://amazon-reviews-2023.github.io/main.html
    """
    
    def __init__(self, root, datasetConfig, datasetName):
        self.root = root
        self.datasetConfig = datasetConfig
        self.datasetName = datasetName
        
        with open(datasetConfig, "r") as configFile:
            configData = json.load(configFile)
            self.datasetConfig = configData.get(datasetName, {})
            print(f"Dataset Config set as:")
            pprint.pp(self.datasetConfig)
        self.category = self.datasetConfig.get("category")
        urls = self.datasetConfig.get("urls", {})
        self.interactionDataUrl = urls.get("interactionDataUrl", "default_interaction_data_url")
        self.metaDataUrl = urls.get("metaDataUrl", "default_meta_data_url")
        self.reviewDataUrl = urls.get("reviewDataUrl", "default_review_data_url")
        
        self.rawDataDir = Path(self.root) / "raw"
        self.rawDataDir.mkdir(parents=True, exist_ok=True)

        self.downloadInteractionData()
        self.downloadItemDataAsJSON()
        self.unwrapItemData(self.rawMetaDatasetPath)
    
    @tryExcept
    @timeMeasured
    def downloadInteractionData(self):
        for split in ["train", "valid", "test"]:
            datasetFilename = f"{self.category}.{split}.csv.gz"
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
    def downloadItemDataFromHF(self):
        datasetFilename = f"{self.category}ItemMetadata.csv.gz"
        datasetPath = self.rawDataDir / "Items" / datasetFilename
        datasetPath.parent.mkdir(parents=True, exist_ok=True)
        if not datasetPath.exists():
            itemInformation = load_dataset(
                "McAuley-Lab/Amazon-Reviews-2023",
                f"raw_meta_{self.category}",
                split="full",
                trust_remote_code=True
            )
            dataframe = pd.DataFrame.from_records(itemInformation)
            dataframe.to_csv(datasetPath, index=False, compression="gzip")
            print(f"File downloaded and saved to {datasetPath}")
    
    def checkRequiredFields(self, jsonLine):
        """
        Checks whether the required fields are present based on the config.
        Returns True if all required fields are valid; False otherwise.
        """
        requiredFields = self.datasetConfig.get("required_fields", {})
        for field, isRequired in requiredFields.items():
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
    def unwrapItemData(self, datasetPath):
        rawUnwrappedItemDataDir = self.rawDataDir / "Items" / "Unwrapped" / self.datasetName
        os.makedirs(rawUnwrappedItemDataDir, exist_ok=True)
        
        with gzip.open(datasetPath, "rt", encoding="utf-8") as f:
            linesCount, self.dumpedJSONlist = 0, []
            for line in f:
                linesCount += 1
                jsonLine = json.loads(line.strip())
                if self.checkRequiredFields(jsonLine):  # Only save if required fields are valid
                    parent_asin = jsonLine.get("parent_asin")
                    outputFilePath = rawUnwrappedItemDataDir / f"{parent_asin}.json"
                    with open(outputFilePath, "w", encoding="utf-8") as outputFile:
                        json.dump(jsonLine, outputFile, indent=4)
                    self.dumpedJSONlist.append(parent_asin)
                if linesCount == 100:
                    break
        
        print(f"Unwrapped dataset from {datasetPath}")
        print(f"Saved {len(self.dumpedJSONlist)} from a total of {linesCount} lines.")

        
        
            
            
if __name__ == "__main__":
    
    root="data/AmazonReviews"
    os.makedirs(root, exist_ok=True)
    datasetConfigAmazon = "src/data/datasetConfigAmazon.json"
    datasetName = "AmazonAllBeautyDataset"
    AmazonBeautyDataset = AmazonDataset(root, datasetConfigAmazon, datasetName)