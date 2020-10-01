'''
NOTES: Loading L1G multi-domain unpaired training data for satellite-to-satellite translation
'''

from petastorm import make_reader, TransformSpec
from petastorm.pytorch import DataLoader

from torchvision import transforms
import torch

import time
import numpy as np

import utils


class _transform_row:
    '''
    Transformations to each row
    Current only performing normalization
    '''
    def __init__(self, sensor):
        mu, sd = utils.get_sensor_stats(sensor)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mu, sd),
                                             ])
    def __call__(self, row):
        data = row['data']   #np.transpose(row['data'], (2,0,1)) 
        return {'data': self.transform(data)}


def make_loaders(params):
    '''
    Data parameters from training configuration file are used to build
        training generators
    Args:
        params
    Returns:
        
    '''
    loaders = dict()
    datanames = params['data'].keys()
    drop_columns = ['year', 'dayofyear', 'hour', 'minute', 'file', 'h', 'v', 'sample_id']
    for key in datanames:
        url = params['data'][key]['data_url']
        transform = TransformSpec(_transform_row(key), removed_fields=drop_columns)
        loaders[key] = DataLoader(make_reader(url, transform_spec=transform), batch_size=params['batch_size'])
    return loaders


def loaders_generator(loaders, bands):
    '''
    Iterates through an 'anchor' loader while sampling from the remaining
    Args:
        loaders: List of dataloaders output from make_loaders
        bands: List (ints) of band 
    Returns:
        Generates a single training s. ample
    '''
    names = list(loaders.keys())
    anchor = names[0]
    others = names[1:]
    iterators = {k: iter(v) for k, v in loaders.items()}
    with loaders[anchor] as loader0:
        for data0 in loader0:
            sample = dict()
            sample[anchor] = data0['data'][:,bands[anchor]]
            # get an example from each of the other loaders
            for key in others:
                try:
                    sample[key] = next(iterators[key])['data'][:,bands[key]]
                except StopIteration:
                    iterators[key] = iter(loaders[key])
                    sample[key] = next(iterators[key])['data'][:,bands[key]]
            yield sample


def make_L1G_generators(params):
    '''
    This builds loaders for training unpaired satellite-to-satellite translation.
    Generates samples [X1, X2,...] from configuation parameters

    Args:
        params: output from configuation file, utils.load_config(config_file)
    Returns:
        generator
    '''
    loaders = make_loaders(params)
    
    datakeys = list(params['data'].keys())
    bands = {d: [int(x) for x in params['data'][d]['bands'].split(',')] for d in datakeys}
    
    generator = loaders_generator(loaders, bands)
    for sample in generator:
        yield sample