import os, sys
import glob
import argparse

import torch

import geonexl1b
import numpy as np
import xarray as xr

import pyhdf.error

from schema import L1bSchema

from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType

from petastorm.unischema import dict_to_spark_row
from petastorm.etl.dataset_metadata import materialize_dataset


def read_file_examples(filepath, patch_size):
    """
    Args:
        filepath: L1b hdf filepath from global gridding system
        patch_size: Size of patches to select from tile
    Return:
        Array of samples selected from tile of shape (N_samples, patch_size, patch_size, 16)
    """
    try:
        x = geonexl1b.L1bFile(filepath, resolution_km=2.).load()
        x[x == 0] = np.nan
        xmin = np.nanmin(x, axis=(0,1))
        xmax = np.nanmax(x, axis=(0,1))
        if xmin[11] < 100:
            print('low value for band 12 in', filepath)
            sys.exit()
    except pyhdf.error.HDF4Error:
        return []

    h, w, c = x.shape
    if patch_size == h:
        return x[np.newaxis]

    r = list(range(0, h, patch_size))
    r[-1] = h - patch_size
    samples = [x[np.newaxis,i:i+patch_size, j:j+patch_size] for i in r for j in r]
    samples = np.concatenate(samples, 0)
    return samples

def random_select_samples(x, ratio=0.1):
    """
    Randomly select indicies from the first index
    """
    N_rand = int(x.shape[0] * ratio)
    rand_idxs = np.random.randint(0, x.shape[0], N_rand)
    return x[rand_idxs]

def sample_generator(x, patch_size=64):
    print(f"Reading file: {x[6]}")
    samples = read_file_examples(x[6], patch_size)
    output = []
    if len(samples) > 0:
        finite_samples = np.all(np.isfinite(samples), axis=(1,2,3))
        samples = samples[finite_samples]
        for j, sample in enumerate(samples):
            output.append({'year': x[0], 'dayofyear': x[1], 
                           'hour': x[2], 'minute': x[3], 
                           'file': x[6], 'data': sample.astype(np.float32), 
                           'sample_id': j, 'h': x[5],
                           'v': x[4]})  
    return output

def generate_dataset(data_directory, sensor, output_url,
                     year=2018, max_files=100000, dayofyear=None):
    """
    Write L1b patches to petastorm database for training
    Args:
        data_directory: directory of L1b data
        sensor: Select sensor from (G16,G17,H8)
        output_url: Directory to write petastorm database (file:///...)
        year: Integer (depending on directory, 2017-2020)
        max_files: Maximum number of files to iterate over
        dayofyear: 1-366
    Returns:
        None
    """
    rowgroup_size_mb = 256

    spark = SparkSession.builder.config('spark.driver.memory', '2g').master('local[4]').getOrCreate()
    sc = spark.sparkContext

    geo = geonexl1b.GeoNEXL1b(data_directory=data_directory, sensor=sensor)
    tiles = geo.tiles()
    files = geo.files(year=year, dayofyear=dayofyear)
    files['v'] = files['tile'].map(lambda t: int(t[4:6]))
    files['h'] = files['tile'].map(lambda t: int(t[1:3]))
    
    idxs = np.random.randint(0, files.shape[0], max_files)
    files = files.iloc[idxs]
    files = files.reset_index()

    with materialize_dataset(spark, output_url, L1bSchema, rowgroup_size_mb):
        filerdd = spark.createDataFrame(files)\
             .select("year", "dayofyear", "hour", "minute", "v", "h", "file")\
             .rdd.map(tuple)\
             .flatMap(sample_generator)\
             .map(lambda x: dict_to_spark_row(L1bSchema, x))

        spark.createDataFrame(filerdd, L1bSchema.as_spark_schema())\
            .coalesce(50) \
            .write \
            .mode('overwrite') \
            .parquet(output_url)

if __name__ == '__main__':
    # python write_data_to_petastorm.py /nex/datapool/geonex/public/GOES16/GEONEX-L1B/ /nobackupp10/tvandal/data/petastorm/G16_64x64_2km/ G16 --year 2018
    # python write_data_to_petastorm.py /nex/datapool/geonex/public/GOES17/GEONEX-L1B/ /nobackupp10/tvandal/data/petastorm/G17_64x64_2km/ G17 --year 2019
    # python write_data_to_petastorm.py /nex/projects/goesscratch/weile/AHI05_java/output/HDF4 /nobackupp10/tvandal/data/petastorm/H8_64x64_2km/ H8 --year 2018

    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('sensor', type=str)
    parser.add_argument('--year', type=int, default=2018)
    parser.add_argument('--max_files', type=int, default=100000)
    args = parser.parse_args()

    output_url = f'file://{args.output_path}'
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    generate_dataset(args.data_path, args.sensor, output_url, max_files=args.max_files, year=args.year)
