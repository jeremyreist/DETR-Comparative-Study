import os
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm
import tarfile
import ssl

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
            # Set a custom user agent
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'}
            req = urllib.request.Request(url, headers=headers)

            # Disable SSL certificate verification
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

            with urllib.request.urlopen(req, context=ctx) as u:
                with open(save_path, 'wb') as f:
                    block_sz = 8192
                    while True:
                        buf = u.read(block_sz)
                        if not buf:
                            break
                        f.write(buf)
                        t.update(len(buf))

        if 'tar' not in save_path:
            print(f"Unzipping dataset to {unzip_path}")
            with zipfile.ZipFile(save_path, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
        else:
            with tarfile.open(save_path, 'r') as tar:
                tar.extractall(unzip_path)

        os.remove(save_path)


# Check and download MOT20 dataset
mot20_url = "https://motchallenge.net/data/MOT20.zip"
mot20_save_path = "data/MOT20/MOT20.zip"
mot20_unzip_path = "data/MOT20/"
if not os.path.exists(mot20_unzip_path):
    download_and_unzip(mot20_url, mot20_save_path, mot20_unzip_path)

# Check and download YouTube-Objects 2.2 "car" dataset
yt_objects_url = "http://calvin-vision.net/bigstuff/youtube-objectsv2/car.tar.gz"
yt_objects_save_path = "data/YouTube-Objects-2.2/car.tar.gz"
yt_objects_unzip_path = "data/YouTube-Objects-2.2/"
if not os.path.exists(yt_objects_unzip_path):
    download_and_unzip(yt_objects_url, yt_objects_save_path, yt_objects_unzip_path)

# Check and download YouTube-Objects 2.2 ground truth
yt_objects_url = "http://calvin-vision.net/bigstuff/youtube-objectsv2/GroundTruth.tar.gz"
yt_objects_save_path = "data/YouTube-Objects-2.2/GroundTruth.tar.gz"
yt_objects_unzip_path = "data/YouTube-Objects-2.2/GroundTruth"
if not os.path.exists(yt_objects_unzip_path):
    download_and_unzip(yt_objects_url, yt_objects_save_path, yt_objects_unzip_path)
