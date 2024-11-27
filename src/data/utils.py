import requests
from datetime import datetime, timezone
import time

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
        
        
"""def timestampToYear(unixTimestamp):
    utcTime = time.gmtime(unixTimestamp)
    return time.strftime("%Y-%m-%d %H:%M:%S", utcTime)"""

"""def timestampToYear(unixTimestamp):
    datetimeObject = datetime.fromtimestamp(unixTimestamp, tz=timezone.utc)
    return datetimeObject.year"""

def timestampToYear(timestamp):
    try:
        # Convert to int and divide by 1000 for millisecond timestamps
        timestamp = int(timestamp) / 1000
        # Convert to datetime and extract the year
        dt_object = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        return dt_object.year
    except (ValueError, TypeError, OverflowError):
        return None