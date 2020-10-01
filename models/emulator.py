import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob

import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import get_sensor_stats, scale_image

from . import layers

class MAIACEmulatorCNN(nn.Module):
    def __init__(self, in_ch):
        super(MAIACEmulatorCNN, self).__init__()
        self.h1 = nn.Conv2d(in_ch, 512, 3, padding=1, stride=1)
        self.concrete1 = layers.ConcreteDropout()
        self.h2 = nn.Conv2d(512, 512, 3, padding=1, stride=1)
        self.concrete2 = layers.ConcreteDropout()
        self.h3 = nn.Conv2d(512, 512, 3, padding=1, stride=1)
        self.concrete3 = layers.ConcreteDropout()
        self.h4 = nn.Conv2d(512, 13, 3, padding=1, stride=1)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, train=True):
        #x1 = self.h1(x)
        x1, reg1 = self.concrete1(x, self.h1)
        x1_1 = self.activation(x1)

        x2, reg2 = self.concrete2(x1_1, self.h2)
        x2_1 = self.activation(x2)

        x3, reg3 = self.concrete3(x2_1, self.h3)
        x3_1 = self.activation(x3)

        x4 = self.h4(x3_1 + x1_1)
        prob = self.sigmoid(x4[:,:1])
        mu = x4[:,1:7]
        logvar = x4[:,7:]
        if train:
            regularizer_loss = reg1 + reg2 + reg3
            return mu, logvar, prob, regularizer_loss
        return mu, logvar, prob

class MAIACTrainer(nn.Module):
    def __init__(self, params):
        super(MAIACTrainer, self).__init__()
        self.params = params

        # set model
        self.model = MAIACEmulatorCNN(params['input_dim'])
        self.checkpoint_filepath = os.path.join(params['model_path'], 'checkpoint.flownet.pth.tar')

        # set optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params['lr'], weight_decay=1e-4)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        self.global_step = 0
        self.tfwriter = SummaryWriter(os.path.join(params['model_path'], 'tfsummary'))

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

    def step(self, x, y, mask, log=False):
        y[mask == 1]  = 0.
        eps = 1e-7
        y_hat, logvar, y_prob, reg_losses = self.model(x, train=True)

        y_cond = torch.masked_select(y, mask==0)
        y_hat_cond = torch.masked_select(y_hat, mask==0)
        y_logvar_cond = torch.masked_select(logvar, mask==0)
        y_precision_cond = torch.exp(-y_logvar_cond) + eps

        cond_logprob = y_precision_cond * (y_cond -  y_hat_cond) ** 2 + y_logvar_cond
        cond_logprob *= -1
        cond_logprob = torch.mean(cond_logprob)

        logprob_classifier = mask * torch.log(y_prob + eps) + (1-mask) * torch.log(1-y_prob + eps)
        logprob_classifier = torch.mean(logprob_classifier)
        logprob = logprob_classifier + cond_logprob

        neglogloss = -logprob

        loss = neglogloss + 1e-2 * reg_losses

        #print('loss', loss)
        #print('null inputs', torch.mean((x != x).type(torch.FloatTensor)))
        if loss != loss:
            print(f"Loss binary: {-logprob_classifier.item()}, Loss Regression: {-cond_logprob.item()}")
            print('x', x[0,0,:,:].cpu().detach().numpy())
            print('y', y[0,0,:,:].cpu().detach().numpy())
            print('regression', y_reg[0,0,:,:].cpu().detach().numpy())
            print('probs', y_prob[0,0,:,:].cpu().detach().numpy())
            print('mask', mask[0,0,:,:].cpu().detach().numpy())
            print('sq_err', sq_err[0,0,:,:].cpu().detach().numpy())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.global_step += 1

        #print('next output', self.model(x)[0])

        if log:
            step = self.global_step
            tfwriter = self.tfwriter
            tfwriter.add_scalar("losses/binary", -logprob_classifier, step)
            tfwriter.add_scalar("losses/regression", -cond_logprob, step)
            tfwriter.add_scalar("losses/regularizer", -reg_losses, step)
            tfwriter.add_scalar("losses/loss", loss, step)

            y_hat *= 1-mask
            # create grid of images
            x_grid = torchvision.utils.make_grid(x[:8,[2,1,0]])
            y_grid = torchvision.utils.make_grid(y[:8,[2,1,0]])
            mask_grid = torchvision.utils.make_grid(mask[:8,:1])

            seg_grid = torchvision.utils.make_grid(y_prob[:8])
            y_reg_grid = torchvision.utils.make_grid(y_hat[:8,[2,1,0]])

            # write to tensorboard
            tfwriter.add_image('inputs', scale_image(x_grid), step)
            tfwriter.add_image('label', scale_image(y_grid), step)
            tfwriter.add_image('regression', scale_image(y_reg_grid), step)

            tfwriter.add_image('mask', mask_grid, step)
            tfwriter.add_image('segmentation', seg_grid, step)
            tfwriter.add_histogram('segmentation', y, step)
            tfwriter.add_histogram('cond_observed', y_cond, step)
            tfwriter.add_histogram('cond_regression', y_hat_cond, step)

        return loss

if __name__ == "__main__":
    params = {'lr': 0.0001, 
              'file_path': '/nobackupp10/tvandal/nex-ai-geo-translation/.tmp/maiac-training-data/',
              'model_path': '/nobackupp10/tvandal/nex-ai-geo-translation/.tmp/models/maiac_emulator/test1/'
             }
    trainer = MAIACTrainer(params)
