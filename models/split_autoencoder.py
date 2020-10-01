import torch
from torch import nn
from torch.autograd import Variable
from .unit_networks import ContentEncoder, Decoder, DecoderSkip, ResBlocks

class SpectralEncoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
                nn.Conv2d(in_ch, 128, 3, padding=0, stride=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 128, 1, padding=0, stride=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, out_ch, 3, padding=0, stride=1),
        )

    def forward(self, x):
        return self.model(x)

class DecoderSmall(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
                nn.Conv2d(in_ch, 64, 3, padding=0, stride=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, 1, padding=0, stride=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, out_ch, 3, padding=0, stride=1),
        )

    def forward(self, x):
        return self.model(x)

class SplitGenVAE(nn.Module):
    def __init__(self, params):
        super(SplitGenVAE, self).__init__()
        self.params = params
        n_downsample = params['gen']['n_downsample']
        n_res = params['gen']['n_res']
        activ = params['gen']['activ']
        pad_type = params['gen']['pad_type']

        if 'skip_connection' not in params['gen'].keys():
            self.skip_dim = None
        elif isinstance(params['gen']['skip_connection'], int):
            self.skip_dim = [params['gen']['skip_connection'],]
        elif isinstance(params['gen']['skip_connection'], str):
            self.skip_dim = [int(i) for i in params['gen']['skip_connection'].split(",")]

        encoders = dict()
        decoders = dict()
        for name, item in params['data'].items():
            enc_dim = params['gen']['dim']
            encoders[name] = ContentEncoder(n_downsample, n_res, item['dim'], enc_dim, 'none', 
                                            activ, pad_type=pad_type)
            enc_dim *= (2 ** n_downsample)
            if self.skip_dim:
                decoders[name] = DecoderSkip(n_downsample, n_res, enc_dim, item['dim'], res_norm='none', 
                                         activ=activ, pad_type=pad_type, skip_dim=self.skip_dim)
            else:
                decoders[name] = Decoder(n_downsample, n_res, enc_dim, item['dim'], res_norm='none', 
                                         activ=activ, pad_type=pad_type)

        self.names = encoders.keys()
        self.encoders = nn.ModuleDict(encoders)
        self.decoders = nn.ModuleDict(decoders)
        self.shared = ResBlocks(1, enc_dim, norm='none', activation='relu')

    def encode(self, x, name):
        enc = self.encoders[name](x)
        enc = self.shared(enc)
        noise = Variable(torch.randn(enc.size()).cuda(enc.data.get_device()))
        return enc, noise

    def decode(self, z, name, skip_x=None):
        if self.skip_dim:
            return self.decoders[name](z, skip_x)
        else:
            return self.decoders[name](z)

    def forward(self, x, name, skip_x=None):
        z, _ = self.encode(x, name)
        return self.decode(z, name, skip_x=skip_x)

    def add_domain(self, name, dim):
        enc_dim = self.params['gen']['dim']
        n_downsample = self.params['gen']['n_downsample']
        n_res = self.params['gen']['n_res']
        activ = self.params['gen']['activ']
        pad_type = self.params['gen']['pad_type']

        encoder = ContentEncoder(n_downsample, n_res, dim, enc_dim, 'none', activ, pad_type=pad_type)
        if self.skip_dim:
            decoder = DecoderSkip(n_downsample, n_res, enc_dim, dim, res_norm='none', activ=activ, pad_type=pad_type)
        else:
            decoder = Decoder(n_downsample, n_res, enc_dim, dim, res_norm='none', activ=activ, pad_type=pad_type)

        self.encoders.update({name: encoder})
        self.decoders.update({name: decoder})
