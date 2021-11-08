import torch 
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import imageio
import numpy as np 
import matplotlib 

from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
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
				transforms.Normalize((0.5,), (0.5,))
])

to_pil_image = transforms.ToPILImage()

train_data = datasets.MNIST(
	root='../input/data',
	train=True,
	transform=transform,
	download=True 
   )
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Generator NN
class Generator(nn.Module): 
	def __init__(self, noise):
		super(Generator, self).__init__()
		self.noise = noise
		self.main = nn.Sequential(
			nn.Linear(self.noise, 256),
			nn.LeakyReLU(0.2),
			
			nn.Linear(256, 512),
			nn.LeakyReLU(0.2),
			
			nn.Linear(512, 1024),
			nn.LeakyReLU(0.2),

			nn.Linear(1024, 784),
			nn.Tanh()
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
			nn.Sigmoid()

		)
	def forward(self, x) : 
		x = x.view(-1, 784)
		return self.main(x)

#init NN's
generator = Generator(noise)
discriminator = Discriminator()

print('### GENERATOR ###')
print(generator)
print('#################')

print('\n### DISCRIMINATOR ###')
print(discriminator)
print('######################')

#Optims
optim_g = optim.Adam(generator.parameters(), lr=0.0002)
optim_d = optim.Adam(discriminator.parameters(), lr=0.0002)

#Loss
criter = nn.BCELoss()

losses_g = []
losses_d = []
images = []

#create real labels
def label_real(size):
	data = torch.ones(size, 1)
	return data
def label_fake(size): 
	data = torch.zeros(size, 1)
	return data

#creating noise vector
def create_noise(sample_size, noise):
	return torch.randn(sample_size, noise)

# Saving images
def save_generator_image(image, path):
	save_image(image, path)

#train discriminator
def train_discriminator(optimizer, data_real, data_fake): 
	b_size = data_real.size(0)
	real_label = label_real(b_size)
	fake_label = label_fake(b_size)

	optimizer.zero_grad()
	
	output_real = discriminator(data_real)
	loss_real = criter(output_real, real_label)
	
	output_fake = discriminator(data_fake)
	loss_fake = criter(output_fake, fake_label)
	
	
	loss_real.backward()
	loss_fake.backward()
	optimizer.step()
	
	return loss_real + loss_fake
 
def train_generator(optimizer, data_fake): 
	b_size = data_fake.size(0)
	real_label = label_real(b_size)
	
	opimizer.zero_grad()
	
	output = discriminator(data_fake)
	loss = criter(output, real_label)
	
	loss.backward()
	optimizer.step()

	return loss

#noise vector
noize = create_noise(sample_size, noise)

generator.train()
discriminator.train()

for epoch in range(epochs): 
	loss_g = 0.0
	loss_d = 0.0
	for bi, data in tqdm(enumerate(train_loader), total=int(len(train_data)/train_loader.batch_size)): 
		image, _ = data
		b_size = len(image)
		#Run discrim for k num steps
		for step in range(k): 
			data_fake = generator(create_noise(b_size, noise)).detach()
			data_real = image
			#descrim train network
			loss_d += train_discriminator(optim_d, data_real, data_fake)
		data_fake = generator(create_noise(b_size, noise))
		
	#final fakey
	generated_img = generator(noize).cpu().detach()
	#make the image grid
	generated_img = make_grid(generated_img)
	#save gen models to disk
	save_generator_image(generated_img, "../outputs/gen_img%d.png" % epoch) 
	images.append(generated_img)
	epoch_loss_g = loss_g / bi
	epoch_loss_d = loss_d / bi
	losses_g.append(epoch_loss_g)
	losses_d.append(epoch_loss_d)

	print("Epoch %d" % epoch)
	print(f"Generator loss: {epoch_loss_g: .8f}, Discriminator loss: {epoch_loss_d: .8f}")

print(' DONE ')
torch.save(generator.state_dict(), '../outputs/generator.pth')

##save gen image as GIF
imgs = [np.array(to_pil_image(image)) for img in images]
imageio.mimsave('../outputs/generator_images.gif', imgs)

#plotting
plt.figure()
plt.plot(losses_d, label='Discriminator loss')
plt.plot(losses_g, label='Generator loss')
plt.legend()
plt.show()



























































