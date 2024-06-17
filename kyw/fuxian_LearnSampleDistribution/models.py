import torch
import torch.nn as nn

from utils import idx2onehot


class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=False, condition_dim=0):

        super().__init__()

        if conditional:
            assert condition_dim > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, condition_dim)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, condition_dim)

    def forward(self, x, c=None):

        if x.dim() > 2:
            x = x.view(-1, 28*28)

        batch_size = x.size(0)

        means, log_var = self.encoder(x, c)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size])
        z = eps * std + means

        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def inference(self, n=1, c=None):

        batch_size = n
        z = torch.randn([batch_size, self.latent_size])

        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, condition_dim):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += condition_dim

        self.MLP = nn.Sequential()

        # for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        #     self.MLP.add_module(
        #         name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
        #     self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
        self.MLP.add_module('L0', nn.Linear(layer_sizes[0], layer_sizes[1], bias=False))
        self.MLP.add_module('A0', nn.ReLU())
        self.MLP.add_module('D0', nn.Dropout(0.5))
        self.MLP.add_module('L1', nn.Linear(layer_sizes[1], layer_sizes[2], bias=False))
        self.MLP.add_module('A1', nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size, bias=False)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size, bias=False)

    def forward(self, x, c=None):

        if self.conditional:
            x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, condition_dim):

        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + condition_dim
        else:
            input_size = latent_size

        # for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
        #     self.MLP.add_module(
        #         name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
        #     if i+1 < len(layer_sizes):
        #         self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
        #     else:
        #         self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())
        self.MLP.add_module('L0', nn.Linear(input_size, layer_sizes[0], bias=False))
        self.MLP.add_module('A0', nn.ReLU())
        self.MLP.add_module('D0', nn.Dropout(0.5))
        self.MLP.add_module('L1', nn.Linear(layer_sizes[0], layer_sizes[1], bias=False))
        self.MLP.add_module('A1', nn.ReLU())
        self.MLP.add_module('O', nn.Linear(layer_sizes[1], layer_sizes[2], bias=False))

    def forward(self, z, c):

        if self.conditional:
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        return x


class CVAE(nn.Module):

    def __init__(self, obs_dim, goal_dim, x_dim, latent_dim):

        super().__init__()

        assert type(obs_dim) == int
        assert type(goal_dim) == int
        assert type(x_dim) == int
        assert type(latent_dim) == int

        self.latent_size = latent_dim

        self.encoder = Encoder(obs_dim, latent_dim)
        self.decoder = Decoder(latent_dim, obs_dim)

        self.MLP = nn.Sequential()
        self.MLP.add_module('L0', nn.Linear(latent_dim + goal_dim, 256))
        self.MLP.add_module('A0', nn.ReLU())
        self.MLP.add_module('D0', nn.Dropout(0.5))
        self.MLP.add_module('L1', nn.Linear(256, 128))
        self.MLP.add_module('A1', nn.ReLU())
        self.MLP.add_module('D1', nn.Dropout(0.5))
        self.MLP.add_module('O', nn.Linear(128, x_dim))

    def forward(self, obs, goal):
        batch_size = goal.size(0)

        mean, log_var = self.encoder(obs)

        std = torch.exp(0.9 * log_var)
        eps = torch.randn([batch_size, self.latent_size])
        z = eps * std + mean

        recon_obs = self.decoder(z)

        input = torch.cat((z, goal), dim=-1)
        x = self.MLP.forward(input)

        return x, recon_obs, z, mean, log_var


    def inference(self, obs, goal, n=1):

        z, log_var = self.encoder(obs)
        recon_obs = self.decoder(z)

        input = torch.cat((z, goal), dim=-1)
        x = self.MLP.forward(input)

        return x, recon_obs


class Encoder(nn.Module):

    def __init__(self, obs_dim, latent_size):
        super().__init__()

        self.MLP = nn.Sequential()
        self.MLP.add_module('L0', nn.Linear(obs_dim, 512))
        self.MLP.add_module('A0', nn.ReLU())
        #self.MLP.add_module('D0', nn.Dropout(0.5))
        self.MLP.add_module('L1', nn.Linear(512, 256))
        self.MLP.add_module('A1', nn.ReLU())
        self.MLP.add_module('L2', nn.Linear(256, 128))
        self.MLP.add_module('A2', nn.ReLU())

        self.linear_means = nn.Linear(128, latent_size)
        self.linear_log_var = nn.Linear(128, latent_size)

    def forward(self, obs):
        recon_obs = self.MLP(obs)
        means = self.linear_means(recon_obs)
        log_vars = self.linear_log_var(recon_obs)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, latent_dim, obs_dim):

        super().__init__()

        self.MLP = nn.Sequential()
        self.MLP.add_module('L0', nn.Linear(latent_dim, 128))
        self.MLP.add_module('A0', nn.ReLU())
        #self.MLP.add_module('D0', nn.Dropout(0.5))
        self.MLP.add_module('L1', nn.Linear(128, 256))
        self.MLP.add_module('A1', nn.ReLU())
        self.MLP.add_module('L2', nn.Linear(256, 512))
        self.MLP.add_module('A2', nn.ReLU())
        self.MLP.add_module('O', nn.Linear(512, obs_dim))

    def forward(self, z):
        recon_obs = self.MLP(z)

        return recon_obs




class Obs2QmapModel(nn.Module):

    def __init__(self, obs_dim, qmap_dim):
        super().__init__()
        latent_dim = 20

        self.encoder = Encoder(obs_dim, latent_dim)
        self.decoder = Decoder(latent_dim, qmap_dim)

    def forward(self, obs):
        z, log_var = self.encoder(obs)
        qmap = self.decoder(z)

        return qmap

    def loss(self, qmap, qmap_target):
        qmap_dim = 56*56
        BCE = torch.nn.functional.mse_loss(
            qmap_target.view(-1, qmap_dim), qmap.view(-1, qmap_dim), reduction='mean')
        return BCE


class Obs2QmapModelwithRecon(nn.Module):

    def __init__(self, obs_dim, qmap_dim):
        super().__init__()
        self.latent_dim = 20

        self.encoder = Encoder(obs_dim, self.latent_dim)
        self.decoder = Decoder(self.latent_dim, obs_dim)
        self.predictor = Decoder(self.latent_dim, qmap_dim)

    def forward(self, obs):
        if len(obs.shape) == 2:
            batch_size = obs.size(0)
        else:
            batch_size = 1

        mean, log_var = self.encoder(obs)
        std = torch.exp(0.9 * log_var)
        eps = torch.randn([batch_size, self.latent_dim])
        z = eps * std + mean

        recon_obs = self.decoder(z)

        qmap = self.predictor(z)

        return qmap, recon_obs, mean, log_var

    def loss(self, qmap, qmap_target, recon_obs, obs, mean, log_var):
        qmap_dim = obs_dim = 56*56

        # qmap预测的损失函数
        qmap_predict_loss = torch.nn.functional.mse_loss(
            qmap_target.view(-1, qmap_dim), qmap.view(-1, qmap_dim), reduction='mean')

        # obs重构损失函数
        BCE = torch.nn.functional.mse_loss(
            recon_obs.view(-1, obs_dim), obs.view(-1, obs_dim), reduction='mean')
        KLD = 10 ** -6 * 2 * torch.mean(log_var.exp() + mean.pow(2) - 1. - log_var, 1)

        recon_obs_loss = torch.mean(BCE + KLD)

        return qmap_predict_loss + recon_obs_loss