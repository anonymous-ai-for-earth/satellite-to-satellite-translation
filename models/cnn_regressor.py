import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob

import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import get_sensor_stats, scale_image

class CNNRegressor(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CNNRegressor, self).__init__()
        self.h1 = nn.Conv2d(in_ch, 256, 5, padding=2, stride=1)
        self.h2 = nn.Conv2d(256, 256, 3, padding=1, stride=1)
        self.h3 = nn.Conv2d(256, 256, 3, padding=1, stride=1)
        self.h4 = nn.Conv2d(256, out_ch, 3, padding=1, stride=1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout2d(0.25)

    def forward(self, x):
        x1 = self.h1(x)
        x1_1 = self.dropout(self.activation(x1))
        x2 = self.h2(x1_1)
        x2_1 = self.dropout(self.activation(x2))
        x3 = self.h3(x2_1)
        x3_1 = self.dropout(self.activation(x3))
        x4 = self.h4(x3_1 + x1_1)
        return x4

class SupervisedCNNTrainer(nn.Module):
    def __init__(self, params):
        super(SupervisedCNNTrainer, self).__init__()
        self.params = params
        # set model
        self.model = CNNRegressor(params['input_dim'], params['output_dim'])
        # set optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params['lr'])
        self.checkpoint_filepath = os.path.join(params['model_path'], 'checkpoint.flownet.pth.tar')
        self.global_step = 0
        self.tfwriter = SummaryWriter(os.path.join(params['model_path'], 'tfsummary'))
        self.mse_loss = nn.MSELoss()

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
        y_reg = self.model(x)
        loss = self.mse_loss(y, y_reg)

        #print('loss', loss)
        #print('null inputs', torch.mean((x != x).type(torch.FloatTensor)))
        if loss != loss:
            print(f"Loss Regression: {loss.item()}")
            print('x', x[0,0,:,:].cpu().detach().numpy())
            print('y', y[0,0,:,:].cpu().detach().numpy())
            print('regression', y_reg[0,0,:,:].cpu().detach().numpy())
            print('probs', y_prob[0,0,:,:].cpu().detach().numpy())
            print('mask', mask[0,0,:,:].cpu().detach().numpy())
            print('sq_err', sq_err[0,0,:,:].cpu().detach().numpy())

        if loss != loss:
            sys.exit()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.global_step += 1

        #print('next output', self.model(x)[0])

        if log:
            step = self.global_step
            tfwriter = self.tfwriter
            tfwriter.add_scalar("losses/loss", loss, step)
            # create grid of images
            x_grid = torchvision.utils.make_grid(x[:,[2,1,0]], nrow=2)
            y_grid = torchvision.utils.make_grid(y[:,[0,]], nrow=2)
            y_reg_grid = torchvision.utils.make_grid(y_reg[:,[0,]], nrow=2)

            # write to tensorboard
            tfwriter.add_image('inputs', scale_image(x_grid), step)
            tfwriter.add_image('label', scale_image(y_grid), step)
            tfwriter.add_image('regression', scale_image(y_reg_grid), step)

        return loss
