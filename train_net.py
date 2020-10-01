'''
Main training script for unsupervised satellite-to-satellite translation.
Performs single node multi-gpu training 

Trains a VAE-GAN for with a shared spectral reconstruction loss
Is applicable to generating virtual sensing but resolution may be lost.
'''

import os, sys
import time
import glob
import argparse

import torch
from torch import nn
from torch.utils import data

import torch.distributed as dist
import torch.multiprocessing as mp
torch.cuda.empty_cache()

import numpy as np
import pandas as pd

from trainer import Trainer
import utils

from data import petastorm_reader

def setup(rank, world_size, port):
    '''
    Setup multi-gpu processing group
    Args:
        rank: current rank
        world_size: number of processes
        port: which port to connect to
    Returns:
        None
    '''
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = f'{port}'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_net_mp(rank, world_size, port, params):
    '''
    Setup and train on node
    '''
    setup(rank, world_size, port)
    train_net(params, rank=rank)

def train_net(params, rank=None, device=None, distribute=True):
    '''
    Run distributed training
    Checkpoints saved to directory params['model_path']
    Tensorboard stats to be found in experiment directory from params['model_path']
    
    Args:
        params: (dict) Parameters from training configuration file
        rank (optional): (int) Process rank, default None
        device (optional): (int) which device to use, default: rank % N_gpus
        distribute (optional): Whether to distribute training
    Returns:
        None
    '''
    if rank == None:
        rank = 0
        setup(rank, 1, 9100+rank)

    print(f"Running training on rank {rank}.")
    # initialize trainer
    if device is None:
        device = rank % torch.cuda.device_count()

    trainer = Trainer(params, distribute=distribute, rank=rank, gpu=device)

    # set device
    if rank == 0:
        trainer.load_checkpoint()

    # Load dataset
    data_generator = petastorm_reader.make_L1G_generators(params)

    t0 = time.time()
    while trainer.global_step < params['max_iter']:
        try:
            for batch_idx, sample in enumerate(data_generator):
                x_dict = {n: x.to(device) for n, x in sample.items()}
                log = False
                if (trainer.global_step % params['log_iter'] == 0) and (rank == 0):
                    log = True
                loss_gen = trainer.gen_update(x_dict, log=log)
                loss_dis = trainer.dis_update(x_dict, log=log)

                if rank == 0:
                    if log:
                        tt = trainer.global_step / (time.time() - t0) * params['batch_size']
                        print(f"Step {trainer.global_step} Examples/Second {tt} -- Generator={loss_gen.item():4.4g}, Discriminator={loss_dis.item():4.4g}")

                    trainer.update_step()
                    if trainer.global_step % params['checkpoint_step'] == 1:
                        trainer.save_checkpoint()

                    if trainer.global_step >= params['max_iter']:
                        break
        finally:
            #loaders = petastorm_reader.make_loaders(params)
            data_generator = petastorm_reader.make_L1G_generators(params)
            
    if rank == 0:
        trainer.save_checkpoint()
    cleanup()

def run_training(params, world_size, port):
    #params['batch_size'] = params['batch_size'] // world_size
    mp.spawn(train_net_mp,
             args=(world_size, port, params,),
             nprocs=world_size,
             join=True)
    cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--port', type=int, default=9002)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    params = utils.get_config(args.config_file)

    run_training(params, args.world_size, args.port)

if __name__ == "__main__":
    main()
