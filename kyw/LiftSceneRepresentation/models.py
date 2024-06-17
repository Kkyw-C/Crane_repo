import torch
import torch.nn as nn


class VAE(nn.Module):

    def __init__(self, x_dim, latent_size, conditional=False, c_dim=0):

        super().__init__()

        self.conditional = conditional

        if self.conditional:
            assert c_dim > 0

        assert type(x_dim) == int
        assert type(latent_size) == int

        self.latent_size = latent_size
        self.c_feature_ex = nn.Sequential(
            # input's shape: (3, 99, 99)
            nn.Conv2d(3, 8, 5, stride=2),   # output's shape ( 8, 48, 48 )
            nn.MaxPool2d(2, 2),             # output's shape ( 8, 24, 24 )
            nn.Conv2d(8, 32, 3),            # output's shape (32, 22, 22 )
            nn.MaxPool2d(2, 2)             # output's shape (32, 11, 11 )
        )
        self.fc = nn.Linear(32*11*11, c_dim)   # output's shape (c_dim)

        self.encoder = Encoder(x_dim, latent_size, conditional, c_dim)
        self.decoder = Decoder(x_dim, latent_size, conditional, c_dim)

    def forward(self, x, c=None):
        batch_size = x.size(0)

        if self.conditional:
            c = self.c_feature_ex(c)
            c = c.view(c.size()[0], -1)  # 相当于keras的flatten（）
            c = torch.nn.functional.relu(self.fc(c))  # c_dim

            means, log_var = self.encoder(x, c)

            std = torch.exp(0.5 * log_var)
            eps = torch.randn([batch_size, self.latent_size])
            z = eps * std + means

            recon_x = self.decoder(z, c)

            return recon_x, means, log_var, z
        else:
            means, log_var = self.encoder(x, c=None)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn([batch_size, self.latent_size])
            z = eps * std + means
            recon_x = self.decoder(z, c=None)

            return recon_x, means, log_var, z


    def inference(self, n=1, c=None):

        batch_size = n
        z = torch.randn([batch_size, self.latent_size])

        if self.conditional:
            c = self.c_feature_ex(c)
            c = c.view(c.size()[0], -1)  # 相当于keras的flatten（）
            c = torch.nn.functional.relu(self.fc(c))  # c_dim

        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, x_dim, latent_size, conditional, condition_dim):

        super().__init__()

        self.conditional = conditional
        input_size = x_dim
        if self.conditional:
            input_size = x_dim + condition_dim

        self.MLP = nn.Sequential()

        self.MLP.add_module('L0', nn.Linear(input_size, 512, bias=False))
        self.MLP.add_module('A0', nn.ReLU())
        self.MLP.add_module('D0', nn.Dropout(0.5))
        self.MLP.add_module('L1', nn.Linear(512, 512, bias=False))
        self.MLP.add_module('A1', nn.ReLU())

        self.linear_means = nn.Linear(512, latent_size, bias=False)
        self.linear_log_var = nn.Linear(512, latent_size, bias=False)

    def forward(self, x, c=None):

        if self.conditional:
            x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, x_dim, latent_size, conditional, condition_dim):

        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + condition_dim
        else:
            input_size = latent_size

        self.MLP.add_module('L0', nn.Linear(input_size, 512, bias=False))
        self.MLP.add_module('A0', nn.ReLU())
        self.MLP.add_module('D0', nn.Dropout(0.5))
        self.MLP.add_module('L1', nn.Linear(512, 512, bias=False))
        self.MLP.add_module('A1', nn.ReLU())
        self.MLP.add_module('O', nn.Linear(512, x_dim, bias=False))

    def forward(self, z, c):

        if self.conditional:
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        return x