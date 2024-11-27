import requests


def loadFileFromURL(url, destinationFilePath):
    """
    Downloads a file from a given URL and saves it to the specified local file path.

    Parameters:
    url (str): The URL of the file to be downloaded.
    localFilePath (str): The local file path (including the file name) where the downloaded file will be saved.
    """
    response = requests.get(url)
    if response.status_code == 200:
        with open(destinationFilePath, 'wb') as f:
            f.write(response.content)
        print(f"File downloaded and saved to {destinationFilePath}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")