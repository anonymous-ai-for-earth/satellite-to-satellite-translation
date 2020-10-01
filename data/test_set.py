import os, sys
import glob
import shutil

import numpy as np
import xarray as xr
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import geonexl1g

def test_set_pair(path1, path2, sensor1, sensor2, year, dayofyear, hour=None, minute=None):
    pair = geonexl1g.L1GPaired(path1, path2, sensor1, sensor2)
    files = pair.files(year=year, dayofyear=dayofyear, how='inner')
    #if hour:
    if isinstance(hour, int):
        files = files[files['hour1'] == hour]
    if isinstance(minute, int):
        files = files[files['minute1'] == minute]
    return files

def copy_file(file, dest):
    if not isinstance(file, str):
        print(f"File is not string: {file}")
        return False
    src_sub = '/'.join(file.split('/')[-6:])
    dest = os.path.join(dest, src_sub)
    dest_dir = os.path.dirname(dest)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    print(f"Copying {file} to {dest}")
    shutil.copy(file, dest)

def sensor_files(path, sensor, year, dayofyear, hour=20, minute=0):
    geo = geonexl1g.GeoNEXL1G(path, sensor)
    files = geo.files(year=year, dayofyear=dayofyear)
    if len(files) == 0:
        return None
    files = files[files['hour'] == hour]
    files = files[files['minute'] == minute]
    return files

def paired_test_set(path1, path2, sensor1, sensor2):
    data_path = '/nobackupp10/tvandal/unsupervised-spectral-synthesis/data/test'
    year = 2019
    days = range(32, 54)
    hour = 0
    file_pairs = []
    for dayofyear in days:
        files = test_set_pair(path1, path2, sensor1, sensor2, year, dayofyear, hour=hour, minute=0)
        print(f"Processing day of year: {dayofyear}")
        file_pairs.append(files.reset_index())
        
    file_pairs = pd.concat(file_pairs)
    return file_pairs

def single_test_set():
    #path = '/nex/datapool/geonex/public/GOES17/GEONEX-L1G/'
    path = '/nex/projects/goesscratch/weile/AHI05_java/output/HDF4'
    sensor = 'H8'
    year = 2019
    dayofyear = 2
    hour = 4
    data_path = '/nobackupp10/tvandal/nex-ai-geo-translation/data/Test/'
    files = sensor_files(path, sensor, year, dayofyear, hour)
    for i, row in files.iterrows():
        print(f"Copying file {row['file']} to {data_path}")
        copy_file(row['file'], data_path)

if __name__ == '__main__':
    #path1 = '/nex/projects/goesscratch/weile/AHI05_java/output/HDF4'
    path1 = '/nex/datapool/geonex/public/GOES16/GEONEX-L1G/'
    path2 = '/nex/datapool/geonex/public/GOES17/GEONEX-L1G/'
    sensor1 = 'G16'
    sensor2 = 'G17'
    paired_file = f'/nobackupp10/tvandal/unsupervised-spectral-synthesis/data/testset_{sensor1}{sensor2}.txt'

    pairs = paired_test_set(path1, path2, sensor1, sensor2)
    print(pairs)
    pairs.to_csv(paired_file, index=False)

    #single_test_set()
