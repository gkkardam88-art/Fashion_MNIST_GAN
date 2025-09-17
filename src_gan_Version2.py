import torch
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from utils import show_imgs

class Discriminator(nn.Module):
    def __init__(self, inp_dim=784):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(inp_dim, 128)
        self.nonlin1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 1)
    def forward(self, x):
        x = x.view(x.size(0), 784)
        h = self.nonlin1(self.fc1(x))
        out = self.fc2(h)
        out = torch.sigmoid(out)
        return out

class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 128)
        self.nonlin1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 784)
    def forward(self, x):
        h = self.nonlin1(self.fc1(x))
        out = self.fc2(h)
        out = torch.tanh(out)
        out = out.view(out.size(0), 1, 28, 28)
        return out

if __name__ == "__main__":
    D = Discriminator()
    G = Generator()
    print("Discriminator:\n", D)
    print("Generator:\n", G)

    z = torch.randn(8, 100)
    samples = G(z)
    show_imgs(samples)