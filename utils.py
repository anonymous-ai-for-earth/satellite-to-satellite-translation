import torch
import torch.nn.init as init

import numpy as np
import cv2
import yaml

def get_sensor_stats(sensor):
    '''
    Values computed by scripts/get_data_stats.py
    inputs
        sensor: (AHI, ABI, G16, G17, H8)
    output
        mean
        standard deviation
    '''
    if sensor in ['AHI', 'H8', 'H8_15']:
        # computed from himawari
        mu = (0.829, 0.512, 0.621, 0.663, 0.427, 0.333, 285.251, 235.759, 244.547,
              252.391, 272.476, 251.321, 274.869, 274.522, 272.611, 259.913)
        sd = (0.292, 0.257, 0.325, 0.362, 0.283, 0.218, 6.243, 2.628, 3.472, 4.165,
              7.574, 4.078, 7.973, 8.122, 7.839, 5.408)
    elif sensor in ['ABI', 'G16', 'G17']:
        mu = (0.829, 0.621, 0.663, 0.077, 0.427, 0.333, 285.251, 235.759, 244.547,
              252.391, 272.476, 251.321, 274.869, 274.522, 272.611, 259.913)
        sd = (0.292, 0.325, 0.362, 0.113, 0.283, 0.218, 6.243, 2.628, 3.472, 4.165,
              7.574, 4.078, 7.973, 8.122, 7.839, 5.408)
    elif sensor in ['AHI12']:
        mu = (0.06, 0.08, 0.15, 0.30, 0.32, 0.23)
        sd = (0.03, 0.1, 0.2, 0.3, 0.3, 0.3)
    return mu, sd

def make_patches(x, patch_size):
    h, w, c = x.shape
    r = list(range(0, h, patch_size))
    r[-1] = h - patch_size
    samples = [x[np.newaxis,i:i+patch_size, j:j+patch_size] for i in r for j in r]
    samples = np.concatenate(samples, 0)
    return samples

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun

def scale_image(x):
    xmn = torch.min(x)
    xmx = torch.max(x)
    return (x - xmn) / (xmx - xmn)

def get_config(yaml_file):
    '''
    Args:
        yaml_file (str): configuration file like configs/Base-G16G18.yaml
    Returns:
        (dict): dictionary of parameters
    '''
    with open(yaml_file) as f:
        return yaml.load(f)
