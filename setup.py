import os
import subprocess
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

# Custom function for tqdm progress bar
class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

# Download and unzip datasets with progress bar
def download_and_unzip(url, save_path, unzip_path):
    if not os.path.exists(unzip_path):
        os.makedirs(unzip_path)

        print(f"Downloading dataset from {url}")
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, save_path, reporthook=t.update_to)

        print(f"Unzipping dataset to {unzip_path}")
        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

        os.remove(save_path)

# Check and download MOT20 dataset
mot20_url = "https://motchallenge.net/data/MOT20.zip"
mot20_save_path = "data/MOT20/MOT20.zip"
mot20_unzip_path = "data/MOT20/MOT20"
if not os.path.exists(mot20_unzip_path):
    download_and_unzip(mot20_url, mot20_save_path, mot20_unzip_path)

# Check and download YouTube-Objects "car" dataset
yt_objects_url = "https://data.vision.ee.ethz.ch/cvl/youtube-objects/categories/car.tar.gz"
yt_objects_save_path = "data/YouTube-Objects/car/car.zip"
yt_objects_unzip_path = "data/YouTube-Objects/car"
if not os.path.exists(yt_objects_unzip_path):
    download_and_unzip(yt_objects_url, yt_objects_save_path, yt_objects_unzip_path)