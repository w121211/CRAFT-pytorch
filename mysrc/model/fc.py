import torch
import torch.nn as nn
import torch.nn.functional as F


class Regressor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Regressor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        """
        Args: x: (N, n_masks, z_dim), features of image
        Return: (N, )
        """
        return self.fc(x)


class Classifier(nn.Module):
    def __init__(self, in_size, n_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_size, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.fc(x)
