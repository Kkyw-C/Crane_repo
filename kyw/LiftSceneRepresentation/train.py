import os
import time
import torch
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import numpy as np
from PIL import Image


from models import VAE


class GoodConfigurationDataset(Dataset):
    """优良位形数据集，我们数据集的样本将是一个字典{'x': conf, 'c': obs, init_CLR, goal_CLR}"""
    def __init__(self, csv_confs_file, init, obs_file, init_CLR_file, goal_CLR_file, transform=None):
        self.confs = np.loadtxt(csv_confs_file, delimiter=',', skiprows=1)  # 跳过第一行的表头
        self.init = init
        self.obs = np.loadtxt(obs_file)
        self.init_CLR = np.loadtxt(init_CLR_file)
        self.goal_CLR = np.loadtxt(goal_CLR_file)
        self.transform = transform



    def __len__(self):
        return len(self.confs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        if self.init == True:
            x = self.confs[idx, 0:3]
        else:
            x = self.confs[idx, 7:10]

        c = np.array([self.obs, self.init_CLR, self.goal_CLR], dtype=np.uint8)
        c = c.transpose(1, 2, 0)

        c = Image.fromarray(np.uint8(c))
        #print(c)



        if self.transform:
            c = self.transform(c)
            #x = x * 99 / 250

        sample = {'x': x, 'c': c}

        return sample



def show_samples(samples, c):
    if len(samples.shape) == 1:
        samples = np.reshape(samples,(1, -1))
        print(len(samples))

    plt.scatter(samples[:, 0], samples[:, 1], color=c, s=7, alpha=0.1)

def show_grid(grid, c):
    x, y = np.nonzero(grid)
    x, y = x - 125, y - 125
    plt.scatter(x, y, color=c, s=7, alpha=0.8)


# trans = transforms.Compose([transforms.Resize(99), transforms.ToTensor()])
# dataset = GoodConfigurationDataset("RRTCon.csv", False, "bin_map.txt", "init_loc_grid.txt", "goal_loc_grid.txt",
#                                        transform=trans)
# #plt.ion()
# plt.figure(figsize=(250.0, 250.0))
# plt.axes().set_aspect('equal')    # 为了绘制的图形不变形
# plt.title('Lifting Scene')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.xlim([-125, 125])  # 设置x轴的边界
# plt.ylim([-125, 125])  # 设置y轴的边界
#
# c = np.array(dataset[:]['c'])*255
# #plt.imshow(c)
# print(dataset[:]['c'])
#
# show_grid(c[:,:, 1], "blue")
# show_grid(c[:,:, 2], "yellow")
# show_samples(dataset[:]['x'], 'green')
# show_grid(c[:,:, 0], "red")
#
# plt.show()




def main(args):

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ts = time.time()

    # dataset = MNIST(
    #     root='data', train=True, transform=transforms.ToTensor(),
    #     download=True)

    trans = transforms.Compose([transforms.Resize(99), transforms.ToTensor()])
    dataset = GoodConfigurationDataset("RRTCon.csv", False, "bin_map.txt", "init_loc_grid.txt", "goal_loc_grid.txt", transform=trans)

    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

    def loss_fn(recon_x, x, mean, log_var):
        # print(recon_x.size())
        # print(x.size())

        w = np.array([[1, 1, 1, 0.5, 0.5, 0.5]])
        w = torch.from_numpy(w).float()
        BCE = torch.nn.functional.mse_loss(
            recon_x.view(-1, 3), x.view(-1, 3), reduction='mean')
        KLD = 10 ** -4 * 2 * torch.mean(log_var.exp() + mean.pow(2) - 1. - log_var, 1)

        vae_loss = torch.mean(BCE + KLD)

        return vae_loss

    vae = VAE(
        x_dim=3,
        latent_size=args.latent_size,
        conditional=args.conditional,
        c_dim=50 if args.conditional else 0).to(device)

    print(vae)

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    logs = defaultdict(list)

    plt_on = False
    #plt.ion()

    for epoch in range(args.epochs):

        #tracker_epoch = defaultdict(lambda: defaultdict(dict))

        for iteration, sample_batched in enumerate(data_loader):     # 逐个batch训练

            x, c = sample_batched['x'].float().to(device), sample_batched['c'].float().to(device)
            # print("x size:", x.size())
            # print("c size:", c.size())

            if args.conditional:
                recon_x, mean, log_var, z = vae(x, c)
            else:
                recon_x, mean, log_var, z = vae(x, c=None)

            # for i, yi in enumerate(y):
            #     id = len(tracker_epoch)
            #     tracker_epoch[id]['x'] = z[i, 0].item()
            #     tracker_epoch[id]['y'] = z[i, 1].item()
            #     tracker_epoch[id]['label'] = yi.item()

            loss = loss_fn(recon_x, x, mean, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs['loss'].append(loss.item())

            if iteration % args.print_every == 0 or iteration == len(data_loader)-1:
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch, args.epochs, iteration, len(data_loader)-1, loss.item()))

                if args.conditional:
                    #c = torch.arange(0, 100).long().unsqueeze(1)
                    x_gen = vae.inference(n=c.size()[0], c=c)
                else:
                    x_gen = vae.inference(n=64)

                x_gen = x_gen.detach().numpy()

                if plt_on:
                    plt.figure(figsize=(250.0, 250.0))
                    plt.axes().set_aspect('equal')  # 为了绘制的图形不变形
                    plt.title('Lifting Scene')
                    plt.xlabel('X')
                    plt.ylabel('Y')
                    plt.xlim([-125, 125])  # 设置x轴的边界
                    plt.ylim([-125, 125])  # 设置y轴的边界
                    show_samples(x_gen, 'green')
                    show_samples(x, 'blue')
                    fig_name = os.path.join(args.model_root, "E{:d}-{:d}-distribution.png".format(epoch, iteration))
                    plt.savefig( fig_name )
                    plt.clf()
                    plt.close('all')
                    #plt.show()

                # if not os.path.exists(os.path.join(args.fig_root, str(ts))):
                #     if not(os.path.exists(os.path.join(args.fig_root))):
                #         os.mkdir(os.path.join(args.fig_root))
                #     os.mkdir(os.path.join(args.fig_root, str(ts)))

                # plt.savefig(
                #     os.path.join(args.fig_root, str(ts),
                #                  "E{:d}I{:d}.png".format(epoch, iteration)),
                #     dpi=300)
                # plt.clf()
                # plt.close('all')

            if iteration % 500 == 0:
                torch.save(vae.state_dict(), os.path.join(args.model_root, "E{:d}-{:d}-vae.pt".format(epoch, iteration)))

        #df = pd.DataFrame.from_dict(tracker_epoch, orient='index')
        # g = sns.lmplot(
        #     x='x', y='y', hue='label', data=df.groupby('label').head(100),
        #     fit_reg=False, legend=True)
        # g.savefig(os.path.join(
        #     args.fig_root, str(ts), "E{:d}-Dist.png".format(epoch)),
        #     dpi=300)

        #torch.save(vae.state_dict(), os.path.join(args.model_root, "E{:d}-vae.pt".format(epoch)))

def test(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vae = VAE(
        x_dim=3,
        latent_size=args.latent_size,
        conditional=args.conditional,
        c_dim=50 if args.conditional else 0).to(device)

    print(vae)

    vae.load_state_dict(torch.load(args.model_file))
    vae.eval()

    trans = transforms.Compose([transforms.Resize(99), transforms.ToTensor()])
    dataset = GoodConfigurationDataset("RRTCon.csv", False, "bin_map.txt", "init_loc_grid.txt", "goal_loc_grid.txt",
                                       transform=trans)
    data_loader = DataLoader(dataset=dataset, batch_size=10000, shuffle=True)

    for iteration, sample_batched in enumerate(data_loader):     # 逐个batch训练
        x, c = sample_batched['x'].float().to(device), sample_batched['c'].float().to(device)
        if args.conditional:
            # c = torch.arange(0, 100).long().unsqueeze(1)
            x_gen = vae.inference(n=c.size()[0], c=c)
        else:
            x_gen = vae.inference(n=x.size()[0])

        x_gen = x_gen.detach().numpy()

        plt.figure(figsize=(250.0, 250.0))
        plt.axes().set_aspect('equal')    # 为了绘制的图形不变形
        plt.title('Lifting Scene')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim([-125, 125])  # 设置x轴的边界
        plt.ylim([-125, 125])  # 设置y轴的边界

        show_samples(x_gen, 'green')
        show_samples(x, 'blue')
        plt.show()






if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--latent_size", type=int, default=2)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--model_root", type=str, default='models')
    parser.add_argument("--conditional", action='store_true', default=False)

    parser.add_argument("--model_file", type=str, default='models/E19-1000-vae.pt')

    args = parser.parse_args()

    #main(args)
    test(args)