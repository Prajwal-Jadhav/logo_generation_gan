import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, Image
from tqdm import tqdm
import os
import csv


# Common config
batch_size  = 64

# Generator config
sample_size = 100    # Random sample size
g_alpha     = 0.01   # LeakyReLU alpha
g_lr        = 1.0e-4 # Learning rate

# Discriminator config
d_alpha     = 0.01   # LeakyReLU alpha
d_lr        = 1.0e-4 # Learning rate

# Define the dataset and preprocessing
transform = transforms.Compose([
    transforms.Resize((108, 108)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# Combine the images and class indices into a single dataset
class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, main_dir):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = os.listdir(main_dir)
        file = open('labels.csv', 'r')

        # Read its contents into a list of lists
        reader = csv.reader(file)
        self.labels = [row for row in reader]

        

    def __getitem__(self, index):
        img_loc = os.path.join(self.main_dir, self.all_imgs[index])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)

        class_index = self.labels[index].index("1")
        return tensor_image, class_index

    def __len__(self):
        return len(self.all_imgs)

labeled_dataset = LabeledDataset("./data")

# Define the dataloader
batch_size = 64
dataloader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=False)

# Coverts conditions into feature vectors
class Condition(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()

        # From one-hot encoding to features: 10 => 784
        self.fc = nn.Sequential(
            nn.Linear(12, 784),
            nn.BatchNorm1d(784),
            nn.LeakyReLU(alpha))
        
    def forward(self, labels: torch.Tensor):
        # One-hot encode labels
        x = F.one_hot(labels, num_classes=12)

        # From Long to Float
        x = x.float()

        # To feature vectors
        return self.fc(x)

# Reshape helper
class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()

        self.shape = shape

    def forward(self, x):
        return x.reshape(-1, *self.shape)

# Generator network
class Generator(nn.Module):
    def __init__(self, sample_size: int, alpha: float):
        super().__init__()

        # sample_size => 784
        self.fc = nn.Sequential(
            nn.Linear(sample_size, 784),
            nn.BatchNorm1d(784),
            nn.LeakyReLU(alpha))

        # 784 => 16 x 7 x 7 
        self.reshape = Reshape(16, 7, 7)

        # 16 x 7 x 7 => 32 x 14 x 14
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(16, 32,
                               kernel_size=5, stride=2, padding=2,
                               output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(alpha))

        # 32 x 14 x 14 => 1 x 28 x 28
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 1,
                               kernel_size=5, stride=2, padding=2,
                               output_padding=1, bias=False),
            nn.Sigmoid())
            
        # Random value sample size
        self.sample_size = sample_size

        # To convert labels into feature vectors
        self.cond = Condition(alpha)

    def forward(self, labels: torch.Tensor):
        # Labels as feature vectors
        c = self.cond(labels)

        # Batch size is the number of labels
        batch_size = len(labels)

        # Generate random inputs
        z = torch.randn(batch_size, self.sample_size)

        # Inputs are the sum of random inputs and label features
        x = self.fc(z)        # => 784
        x = self.reshape(x+c) # => 16 x 7 x 7
        x = self.conv1(x)     # => 32 x 14 x 14
        x = self.conv2(x)     # => 1 x 28 x 28
        return x

# Discriminator network
class Discriminator(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        
        # 1 x 28 x 28 => 32 x 14 x 14
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32,
                      kernel_size=5, stride=2, padding=2, bias=False),
            nn.LeakyReLU(alpha))

        # 32 x 14 x 14 => 16 x 7 x 7
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 16, 
                      kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(alpha))

        # 16 x 7 x 7 => 784
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 784),
            nn.BatchNorm1d(784),
            nn.LeakyReLU(alpha),
            nn.Linear(784, 1))

        # Reshape label features: 784 => 16 x 7 x 7 
        self.cond = nn.Sequential(
            Condition(alpha),
            Reshape(16, 7, 7))

    def forward(self, images: torch.Tensor,
                      labels: torch.Tensor,
                      targets: torch.Tensor):
        # Label features
        c = self.cond(labels)

        # Image features + Label features => real or fake?
        x = self.conv1(images)    # => 32 x 14 x 14
        x = self.conv2(x)         # => 16 x 7 x 7
        prediction = self.fc(x+c) # => 1

        loss = F.binary_cross_entropy_with_logits(prediction, targets)
        return loss

# To save grid images
def save_image_grid(epoch: int, images: torch.Tensor, ncol: int):
    image_grid = make_grid(images, ncol)     # Into a grid
    image_grid = image_grid.permute(1, 2, 0) # Channel to last
    image_grid = image_grid.cpu().numpy()    # Into Numpy

    plt.imshow(image_grid)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'generated_{epoch:03d}.jpg')
    plt.close()

# Real / Fake targets
real_targets = torch.ones(batch_size, 1)
fake_targets = torch.zeros(batch_size, 1)

# Generator and discriminator
generator = Generator(sample_size, g_alpha)
discriminator = Discriminator(d_alpha)

# Optimizers
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=d_lr)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=g_lr)

# Train loop
for epoch in range(100):

    d_losses = []
    g_losses = []

    for i, (images, labels) in enumerate(dataloader):

        #===============================
        # Disciminator Network Training
        #===============================

        # Images from MNIST are considered as real
        d_loss = discriminator(images, labels, real_targets)
       
        # Images from Generator are considered as fake
        d_loss += discriminator(generator(labels), labels, fake_targets)

        # Discriminator paramter update
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        #===============================
        # Generator Network Training
        #===============================

        # Images from Generator should be as real as ones from MNIST
        g_loss = discriminator(generator(labels), labels, real_targets)

        # Generator parameter update
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # Keep losses for logging
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())
        
    # Print loss
    print(epoch, np.mean(d_losses), np.mean(g_losses))

    # Save generated images
    labels = torch.LongTensor(list(range(12))).repeat(8).flatten()
    save_image_grid(epoch, generator(labels), ncol=12)