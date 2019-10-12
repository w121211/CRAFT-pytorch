import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEEncoder(nn.Module):
    def __init__(self, opt):
        super(VAEEncoder, self).__init__()

        self.n_masks = opt.n_masks
        self.conv = nn.Sequential(
            nn.Conv2d(3 + opt.n_masks, 16, 4, 2, 1),  # 16,32,32
            # nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1),  # 32,16,16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),  # 32,16,16 -> 64,8,8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  # 64,8,8 -> 128,4,4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.AvgPool2d(kernel_size=2),
        )

        out_size = opt.imsize // 2 ** 4
        self.fc = nn.Linear(128 * out_size ** 2, opt.n_masks * opt.z_dim)

    def forward(self, x):
        N = x.size(0)
        x = self.conv(x).view(N, -1)
        x = self.fc(x).view(N, self.n_masks, -1)
        # mu = self.fc1(x)
        # # logvar = self.fc2(x)
        # # std = torch.exp(0.5 * logvar)
        # # eps = torch.randn_like(std)
        # z = mu + eps * std
        #         return z, mu, logvar
        return x


class VAEDecoder(nn.Module):
    def __init__(self, opt):
        super(VAEDecoder, self).__init__()
        # self.fc = nn.Sequential(
        #     nn.Linear(opt.z_dim, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, 4096),
        #     nn.ReLU(inplace=True),
        # )
        # self.conv = nn.Sequential(
        #     nn.Conv2d(16, 16, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        #     nn.Conv2d(16, 1, kernel_size=3, padding=1),
        #     nn.Sigmoid(),
        # )
        self.init_size = opt.imsize // 4
        self.l1 = nn.Sequential(nn.Linear(opt.z_dim, 128 * self.init_size ** 2))

        self.conv = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.im_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        N = z.shape[0]
        x = self.l1(z).view(N, 128, self.init_size, self.init_size)
        x = self.conv(x)
        return x


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        # self.opt = opt
        self.imsize = opt.imsize
        self.device = opt.device
        self.n_masks = opt.n_masks

        self.model = nn.Sequential(
            nn.Linear(opt.z_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, opt.z_dim),
            nn.Linear(opt.z_dim, 3),  # to RGB
            nn.Tanh(),
        )
        # self.blur = GaussianSmoothing(3, 5, 5)

    def forward(self, features, x_img):
        """
        Args:
            features: (N, n_masks, z_dim)
            x_img: (N, n_masks+3, imsize, imsize)
        """
        N = features.shape[0]
        x = self.model(features)  # (N, n_masks, 3)
        rgb = x.view(N, self.n_masks, 3, 1, 1)

        bg = x_img[:, -3:, :, :]
        for i in range(self.n_masks):
            layer = rgb[:, i, :, :, :] * torch.ones(N, 1, self.imsize, self.imsize).to(
                self.device
            )
            mask = x_img[:, i : i + 1, :, :]
            bg = mask * layer + (1.0 - mask) * bg

        return bg

