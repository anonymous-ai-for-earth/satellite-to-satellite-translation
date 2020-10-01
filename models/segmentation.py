import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob

import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import get_sensor_stats, scale_image
from . import unet

class SegCNN(nn.Module):
    def __init__(self, in_ch):
        super(SegCNN, self).__init__()
        self.h1 = nn.Conv2d(in_ch, 32, 5, padding=2, stride=1)
        self.h2 = nn.Conv2d(32, 32, 3, padding=1, stride=1)
        self.h3 = nn.Conv2d(32, 1, 3, padding=1, stride=1)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.h1(x)
        x1_1 = self.activation(x1)
        x2 = self.h2(x1_1)
        x2_1 = self.activation(x2)
        x3 = self.h3(x2_1 + x1_1)
        y = self.sigmoid(x3)
        return y

class SegTrainer(nn.Module):
    def __init__(self, params):
        super(SegTrainer, self).__init__()

        self.params = params

        # set model
        #self.model = SegCNN(16)
        self.model = unet.UNet(4, 1)
        
        # set optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params['lr'])
        self.checkpoint_filepath = os.path.join(params['model_path'], 'checkpoint.flownet.pth.tar')
        self.global_step = 0
        
        self.tfwriter = SummaryWriter(os.path.join(params['model_path'], 'tfsummary'))
        
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
    def load_checkpoint(self):
        filename = self.checkpoint_filepath
        if os.path.isfile(filename):
            print("loading checkpoint %s" % filename)
            checkpoint = torch.load(filename)
            self.global_step = checkpoint['global_step']
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (Step {})"
                    .format(filename, self.global_step))
        else:
            print("=> no checkpoint found at '{}'".format(filename))
        
        
    def save_checkpoint(self):
        state = {'global_step': self.global_step, 
                 'model': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(state, self.checkpoint_filepath)

    def step(self, x, y, log=False):
        y_prob = self.model(x)

        loss = self.bce_loss(y_prob, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.global_step += 1

        if log:
            step = self.global_step
            tfwriter = self.tfwriter
            tfwriter.add_scalar("losses/loss", loss, step)
            
            # create grid of images
            x_grid = torchvision.utils.make_grid(x[:,[2,1,0]])
            y_grid = torchvision.utils.make_grid(y)

            seg_grid = torchvision.utils.make_grid(y_prob)

            # write to tensorboard
            tfwriter.add_image('inputs', scale_image(x_grid), step)
            tfwriter.add_image('labels', y_grid, step)

            tfwriter.add_image('segmentation', seg_grid, step)
            tfwriter.add_histogram('segmentation', y, step)
        return loss


if __name__ == "__main__":
    params = {'lr': 0.0001, 
              'file_path': '/nobackupp10/tvandal/nex-ai-geo-translation/.tmp/maiac-training-data/',
              'model_path': '/nobackupp10/tvandal/nex-ai-geo-translation/.tmp/models/maiac_emulator/test1/'
             }
    trainer = MAIACTrainer(params)