import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import copy
from torch.utils.tensorboard import SummaryWriter

from models import SplitGenVAE
from models.discriminators import Discriminators

import utils

class Trainer(nn.Module):
    def __init__(self, params, distribute=False, rank=0, gpu=0):
        super(Trainer, self).__init__()
        self.params = params
        self.checkpoint_filepath = os.path.join(params['model_path'], 'checkpoint.flownet.pth.tar')
        self.distribute = distribute

        data_names = params['data'].keys()

        comma_to_list = lambda arr: [int(i) for i in arr.split(',')]
        self.shared = {d: {n: comma_to_list(el) for n, el in params['data'][d]['shared'].items()} for d in data_names if 'shared' in params['data'][d]}

        # Set generator 
        self.gen = SplitGenVAE(params) 

        # Set discriminator
        self.dis = Discriminators(params)
        
        if distribute:
            self.gen = DDP(self.gen.to(gpu), device_ids=[gpu])
            self.dis = DDP(self.dis.to(gpu), device_ids=[gpu])
        else:
            self.gen = self.gen.to(gpu)
            self.dis = self.dis.to(gpu)
        
        self.apply(utils.weights_init(params['init']))
        self.dis.apply(utils.weights_init('gaussian'))

        # Define loss
        self.mse_loss = nn.MSELoss()
        self.global_step = 0
        self.cycles = 1
        self.cycle_step = params['cycle_step']

        # Load optimizer
        self.optimizer_gen = torch.optim.Adam([p for p in self.gen.parameters() if p.requires_grad],
                                              lr=params['lr'],
                                              betas=(params['beta1'], params['beta2']),
                                              weight_decay=params['weight_decay'])
        self.optimizer_dis = torch.optim.Adam([p for p in self.dis.parameters() if p.requires_grad],
                                              lr=params['lr'],
                                              betas=(params['beta1'], params['beta2']),
                                              weight_decay=params['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_gen,
                                                         step_size=params['step_size'],
                                                         gamma=params['gamma'])
        #self.load_checkpoint()  # load after setting device

        # Tensorboard writer
        if rank == 0:
            self.tfwriter = SummaryWriter(os.path.join(params['model_path'], 'tfsummary'))

    def add_domain(self, name, dim):
        if self.distribute:
            self.gen.module.add_domain(name, dim)
            self.dis.module.add_domain(name, dim)
        else:
            self.gen.add_domain(name, dim)
            self.dis.add_domain(name, dim)

        comma_to_list = lambda arr: [int(i) for i in arr.split(',')]
        curr_keys = copy.deepcopy(list(self.shared.keys()))
        self.shared[name] = {n: comma_to_list(el) for n, el in self.params['data'][name]['shared'].items()}
        for k in curr_keys:
            self.shared[k][name] = comma_to_list(self.params['data'][k]['shared'][name])

    def save_checkpoint(self):
        if self.distribute:
            state = {'global_step': self.global_step, 'gen_state': self.gen.module.state_dict(),
                     'optimizer_gen': self.optimizer_gen.state_dict(), 'scheduler': self.scheduler.state_dict(),
                     'dis_state': self.dis.module.state_dict(), 'optimizer_dis': self.optimizer_dis.state_dict()}
        else:
            state = {'global_step': self.global_step, 'gen_state': self.gen.state_dict(),
                     'optimizer_gen': self.optimizer_gen.state_dict(), 'scheduler': self.scheduler.state_dict(),
                     'dis_state': self.dis.state_dict(), 'optimizer_dis': self.optimizer_dis.state_dict()}
        torch.save(state, self.checkpoint_filepath)

    def load_checkpoint(self):
        filename = self.checkpoint_filepath
        if os.path.isfile(filename):
            print("loading checkpoint %s" % filename)
            checkpoint = torch.load(filename)
            self.global_step = checkpoint['global_step']
            if self.distribute:
                self.gen.module.load_state_dict(checkpoint['gen_state'])
                self.dis.module.load_state_dict(checkpoint['dis_state'])
            else:
                self.gen.load_state_dict(checkpoint['gen_state'])
                self.dis.load_state_dict(checkpoint['dis_state'])
                
            self.optimizer_gen.load_state_dict(checkpoint['optimizer_gen'])
            self.optimizer_dis.load_state_dict(checkpoint['optimizer_dis'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (Step {})"
                    .format(filename, self.global_step))
        else:
            print("=> no checkpoint found at '{}'".format(filename))

    def update_step(self):
        self.global_step += 1

    def _compute_kl(self, x):
        return torch.mean(x**2)

    def gen_update(self, x_dict, log=False):
        keys = list(x_dict.keys())

        if self.distribute:
            gen = self.gen.module
            dis = self.dis.module
        else:
            gen = self.gen
            dis = self.dis

        z_dict = {name: gen.encode(x, name) for name, x in x_dict.items()}

        if gen.skip_dim:
            x_skip = {name: x[:,gen.skip_dim]  for name, x in x_dict.items()}
        else:
            x_skip = {name: None for name in x_dict.keys()}

        x_recon = {name: gen.decode(z+noise, name, skip_x=x_skip[name]) for name, (z, noise) in z_dict.items()}

        key0 = keys[0]
        
        _, _, h_hat, w_hat = x_recon[key0].shape

        # decode all cross pairs
        loss_gan = 0.
        data_losses = dict()
        loss_shared_recon = 0.
        loss_cycle_z = 0.
        loss_cycle_recon = 0.
        loss_cycle_z_recon = 0.
        cycles = 1 + self.global_step // self.cycle_step

        gan_losses = []
        cycle_recon_losses = []
        shared_losses = []
        cycle_zkl_losses = []
        cycle_zrecon_losses = []
        
        
        for k in keys:
            x_k = x_dict[k]
            z_k, n_k = z_dict[k]
            
            cross_keys = list(gen.decoders.keys()) #copy.copy(keys)
            cross_keys.remove(k)
            
            cross_recon = {kk: gen.decode(z_k + n_k, kk, skip_x=x_skip[k]) for kk in cross_keys}            
            
            # GAN Loss
            for kk in cross_keys:
                gan_losses.append(dis.models[kk].calc_gen_loss(cross_recon[kk]))

            # Reconstruction loss
            data_losses[k] = self.mse_loss(x_k, x_recon[k])

            # Shared reconstruction loss
            if hasattr(self, 'shared') and (k in self.shared):
                shared_losses = [self.mse_loss(x_k[:,self.shared[k][kk]], cross_recon[kk][:,self.shared[kk][k]]) for kk in cross_keys]
            xk_1 = x_k
            zk_1 = z_k

            for c in range(cycles):
                xk_cycle = {kk: gen.decode(zk_1 + n_k, kk, skip_x=x_skip[k]) for kk in keys}            # decode each domain
                z_cycle = {kk: gen.encode(_x, kk) for kk, _x in xk_cycle.items()}                       # encode back to z
                x2_cycle = {kk: gen.decode(z_cycle[kk][0] + n_k, k, skip_x=x_skip[k]) for kk in keys}   # decode back to cross domains

                cycle_recon_losses += [self.mse_loss(x_k, x2_cycle[kk]) for kk in keys]
                cycle_zrecon_losses = [self.mse_loss(z_k, z_cycle[kk][0]) for kk in keys]
                if log and (len(cross_keys) > 0):
                    kshow = cross_keys[0]
                    #self.tfwriter.add_image(f"images/cross_{c}_recon_{kshow}", utils.scale_image(x2_cycle[kshow][0,:1]), self.global_step)
                    #self.tfwriter.add_histogram(f"channel0/cross_reconstruct_{kshow}", x2_cycle[kshow][0,0], self.global_step)
                
                cycle_zkl_losses = [self._compute_kl(z2) for name, (z2, _) in z_cycle.items()]
                
                if c > 0:
                    x1_k = x2_cycle[k]
                    zk_1, _ = gen.encode(x1_k, k)
                    #z1_cycle = {name: self.gen.encode(x, name) for name, x in x2_cycle.items()}


        loss_recon = torch.mean(torch.stack([x for _, x in data_losses.items()]))
        loss_gan = torch.mean(torch.stack(gan_losses))
        loss_cycle_recon = cycles * torch.mean(torch.stack(cycle_recon_losses))
        if len(shared_losses) > 0:
            loss_shared_recon += torch.mean(torch.stack(shared_losses))
        loss_cycle_z = cycles * torch.mean(torch.stack(cycle_zkl_losses))
        loss_cycle_z_recon = cycles * torch.mean(torch.stack(cycle_zrecon_losses))
        
        loss = self.params['gan_w'] * loss_gan + \
               self.params['recon_x_w'] * loss_recon + \
               self.params['recon_x_cyc_w'] * loss_cycle_recon + \
               self.params['recon_z_cyc_w'] * loss_cycle_z_recon + \
               self.params['recon_kl_cyc_w'] * loss_cycle_z + \
               self.params['recon_shared_w'] * loss_shared_recon

        self.optimizer_gen.zero_grad()
        loss.backward()
        self.optimizer_gen.step()
        self.scheduler.step()

        step = self.global_step
        
        if log:
            tfwriter = self.tfwriter
            #print(f"Gen: Step {self.global_step} -- Loss={loss.item()}")
            tfwriter.add_scalar("losses/gen/total", loss, step)
            tfwriter.add_scalar("losses/gen/recon", loss_recon, step)
            tfwriter.add_scalar("losses/gen/gan", loss_gan, step)
            tfwriter.add_scalar("losses/gen/cycle_recon", loss_cycle_recon, step)
            tfwriter.add_scalar("losses/gen/shared_recon", loss_shared_recon, step)
            #tfwriter.add_scalar("losses/gen/cycle_kl", loss_cc_z, step)
            tfwriter.add_scalar("losses/gen/cycle_z_recon", loss_cycle_z_recon, step)

            for name in x_dict:
                x = x_dict[name]
                self.tfwriter.add_image(f"images/{name}/input", utils.scale_image(x[0,:1]), step)
                self.tfwriter.add_image(f"images/{name}/reconstruction", utils.scale_image(x_recon[name][0,:1]), step)
                self.tfwriter.add_scalar(f"losses/data/{name}", data_losses[name], step)                
                for j in range(x.shape[1]):
                    self.tfwriter.add_histogram(f"channel{j}/Observed_Domain{name}", x[:,j], step)
                    self.tfwriter.add_histogram(f"channel{j}/reconstructed_Domain{name}", x_recon[name][:,j], step)
        return loss

    def dis_update(self, x_dict, log=False):
        if self.distribute:
            gen = self.gen.module
            dis = self.dis.module
        else:
            gen = self.gen
            dis = self.dis

        if gen.skip_dim:
            x_skip = {name: x[:,gen.skip_dim]  for name, x in x_dict.items()}
        else:
            x_skip = {name: None for name in x_dict.keys()}

        z_dict = {name: gen.encode(x, name) for name, x in x_dict.items()}
        keys = list(x_dict.keys())

        # for each discriminator, show it examples of true examples + all other false examples        
        dis_losses = []
        for name, x in x_dict.items():
            cross_keys = copy.copy(list(gen.decoders.keys()))
            cross_keys.remove(name)
            x_cross = {kk: gen.decode(z_dict[kk][0] + z_dict[kk][1], name, skip_x=x_skip[kk]) for kk in z_dict.keys()}
            dis_losses += [dis.models[name].calc_dis_loss(xc.detach(), x_dict[name]) for kk, xc in x_cross.items()]
        
        loss_dis = torch.mean(torch.stack(dis_losses))
        loss = loss_dis * self.params['gan_w']

        self.optimizer_dis.zero_grad()
        loss.backward()
        self.optimizer_dis.step()

        if log:
            tfwriter = self.tfwriter
            tfwriter.add_scalar('losses/dis/total', loss_dis, self.global_step)
            
            x_12 = gen.decode(z_dict[keys[0]][0], keys[1], skip_x=x_skip[keys[0]])
            for c in range(x_12.shape[1]):
                tfwriter.add_image(f"images/cross_reconstruction/channel_{c}", utils.scale_image(x_12[0,c:c+1]),self.global_step)
                
        return loss
