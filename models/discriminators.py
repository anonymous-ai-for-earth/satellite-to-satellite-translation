import torch
import torch.nn as nn

from models import MsImageDis

class Discriminators(nn.Module):
    '''
    Builds a dictionary of discriminators for GAN learning of each satellite
    '''
    def __init__(self, params):
        super(Discriminators, self).__init__()
        self.models = nn.ModuleDict({n: MsImageDis(item['dim'], params['dis']) for n, item in params['data'].items()})
        self.params = params
        
    def forward(self, x, name):
        return self.models[name](x)
    
    def add_domain(self, name, dim):
        self.models.update({name: MsImageDis(dim, self.params['dis'])})