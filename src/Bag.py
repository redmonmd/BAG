import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import imageio
import numpy as np 
import matplotlib 

from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
From matplotlib import pyplot as plt
from tqdm import tqdm

matplotlib.style.use('ggplot')

# Learning Params
batch_size =  512
epochs = 250
sample_size = 64
noise = 128
k = 1

transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize((0.5,),(0.5)),
])

to_pil_image = transforms.ToPILImage()

train_data = datasets.MNIST(
	root='../input/data'
	train=True
	Download=True
	transform=transform
   )
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Generator NN
class Generator(nn.Module) : 
	def __init__(self, noise) :
		super(Generator, self).init()
		self.noise = noise
		self.main = nn.Sequential (
			nn.Linear(self.noise, 256),
			nn.LeakyReLU(0.2),
			
			nn.Linear(256, 512),
			nn.LeakyReLU(0.2),
			
			nn.Linear(512, 1024),
			nn.LeakyReLU(0.2),

			nn.Linear(1024, 784),
			nn.Tanh(),
		)

	def forward(self, x) : 
		return self.main(x).view(-1,1,28,28)

class Discriminator(nn.Module) : 
	def __init__(self): 
		super(Discriminator, self).__init__()
		self.n_input = 784
		self.main = nn.Sequential (
			nn.Linear(self.n_input, 1024),
			nn.LeakyReLU(0.2),
			nn.Dropout(0.3),

			nn.Linear(1024, 512),
			nn.LeakyReLU(0.2),
			nn.Dropout(0.3),

			nn.Linear(512, 256),
			nn.LeakyReLU(0.2),
			nn.Dropout(0.3),

			nn.Linear(256, 1),
			nn.Sigmoid(),

		)
	def forward(self, x) : 
		x = x.view(-1, 784)
		return self.main(x)

