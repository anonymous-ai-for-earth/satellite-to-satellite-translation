import os

import numpy as np
import torch

from models import SplitGenVAE
import utils

def domain_to_domain(model, data, domain1, domain2, bands1=None, bands2=None, device=None,
                     latent=False):
    '''
    Perform a forward pass to translate from domain1 to domain2

    Args:
        model (SplitGenVAE): pytorch module
        data (np.array): Array of shape (H,W,C)
        domain1 (str): Name of data domain (G16,G17,H8)
        domain2 (str): Name of target domain (G16,G17,H8)
        bands1 (list or np.array): Indices of bands to select as inputs
        bands2 (list or np.array): Indices of bands to select as outputs
        device (str): which device to load data into
        latent (boolean): Whether to return latent features, default False
    Returns:
        np.array of target prediction
        (optional) np.array of latent features
    '''
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get statistics for normalization
    mu1, std1 = utils.get_sensor_stats(domain1)
    mu2, std2 = utils.get_sensor_stats(domain2)

    # select bands
    if bands1 is None:
        bands1 = np.arange(0,16)
    if bands2 is None:
        bands2 = np.arange(0,16)

    data = data[:,:,bands1]
    mu1 = np.array(mu1)[bands1]
    std1 = np.array(std1)[bands1]

    mu2 = np.array(mu2)[bands2]
    std2 = np.array(std2)[bands2]

    # normalize inputs, add axis, send to device
    data_norm = (data - mu1) / std1
    data_norm = np.transpose(data_norm, (2,0,1))[np.newaxis]
    data_norm = torch.Tensor(data_norm).to(device)

    # encode to latent space
    z, noise = model.encode(data_norm, domain1)
    
    skip_x = None
    if model.skip_dim:
        skip_x = data_norm[:,model.skip_dim]
    
    # decode to target domain
    estimate = model.decode(z, domain2, skip_x=skip_x)[0].detach().cpu().numpy() # decode to bands
    estimate = np.transpose(estimate, (1,2,0))
    # de-normalize target
    estimate = estimate * std2 + mu2

    # place some restrictions on thermal ranges
    thermal_bands = bands2[bands2 >= 6]
    thermal = estimate[:,:,thermal_bands]
    thermal[thermal < 180] = np.nan
    thermal[thermal > 350] = np.nan
    estimate[:,:,thermal_bands] = thermal

    if latent:
        return estimate, z.detach().cpu().numpy()[0]
    return estimate

def load_model(config_file, device=None):
    '''
    Load SplitGenVAE model for inference
    Args:
        config_file: get parameters and model_directory from configuration file
        device: set device for inference
    Return:
        SplitGenVAE (torch.nn.Module)
        params (dict)
    '''
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = utils.get_config(config_file)

    model = SplitGenVAE(params)    
    model.to(device)
    checkpoint_path = os.path.join(params['model_path'], 'checkpoint.flownet.pth.tar')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['gen_state'])
    step = checkpoint['global_step']
    print(f"Loaded model from step: {step}")
    return model, params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('inputfile', type=str)
    parser.add_argument('outputfile', type=str)
    args = parser.parse_args()
    
    model = load_model(args['config'])