import torch.nn as nn
import torch
import numpy as np
from torch import optim

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.utils import Image
from torch.utils import data
import numpy as np
import pandas as pd
import math
import os
import csv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the dataset and preprocessing
transform = transforms.Compose([
    transforms.Resize((108, 108)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the one-hot encoded labels from the CSV file
labels_df = pd.read_csv('./labels.csv', header=None)
labels = torch.from_numpy(np.array(labels_df.values)).float()

# Convert the one-hot encoded labels to class indices
class_indices = torch.argmax(labels, dim=0)

# Combine the images and class indices into a single dataset
class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, main_dir, class_indices):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = os.listdir(main_dir)
        self.class_indices = class_indices
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

labeled_dataset = LabeledDataset("./data", class_indices)

# Define the dataloader
batch_size = 64
dataloader = data.DataLoader(labeled_dataset, batch_size=batch_size, shuffle=False)

class CGANGenerator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_shape):
        super(CGANGenerator, self).__init__()

        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

        self.img_shape = img_shape

    def forward(self, noise, labels):
        # Concatenate noise and labels
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        
        return img

class CGANDiscriminator(nn.Module):
    def __init__(self, num_classes=10):
        super(CGANDiscriminator, self).__init__()
        self.num_classes = num_classes
        
        # input layer for image tensor
        self.image_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # input layer for class label tensor
        self.label_layer = nn.Sequential(
            nn.Embedding(self.num_classes, 512),
            nn.Linear(512, 64*64)
        )
        
        # output layers for image and label tensors
        self.output_layer = nn.Sequential(
            nn.Conv2d(512+64, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )
        
    def forward(self, image, class_index):
        image_features = self.image_layer(image)
        class_embed = self.label_layer(class_index).view(-1, 1, 64, 64)
        print(image_features.shape)
        print(class_embed.shape)
        combined = torch.cat((image_features, class_embed), dim=1)
        output = self.output_layer(combined)
        return output.view(-1, 1).squeeze(1)



def train_cgan(generator, discriminator, dataloader, device, num_classes, latent_dim, num_epochs=100, log_interval=100):
    # Initialize optimizers
    generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Initialize loss function
    adversarial_loss = nn.BCELoss()
    
    # Train the CGAN
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(dataloader):
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)
            class_indices = torch.argmax(labels, dim=0)
            
            # Train discriminator
            discriminator.zero_grad()
            real_outputs = discriminator(images, class_indices)
            real_targets = torch.ones(images.size(0), device=device)
            real_loss = adversarial_loss(real_outputs, real_targets)
            real_loss.backward()
            
            z = torch.randn(images.size(0), latent_dim, device=device)
            fake_images = generator(z, class_indices)
            fake_outputs = discriminator(fake_images.detach(), class_indices)
            fake_targets = torch.zeros(images.size(0), device=device)
            fake_loss = adversarial_loss(fake_outputs, fake_targets)
            fake_loss.backward()
            
            discriminator_optimizer.step()
            
            # Train generator
            generator.zero_grad()
            z = torch.randn(images.size(0), latent_dim, device=device)
            fake_images = generator(z, class_indices)
            fake_outputs = discriminator(fake_images, class_indices)
            real_targets = torch.ones(images.size(0), device=device)
            generator_loss = adversarial_loss(fake_outputs, real_targets)
            generator_loss.backward()
            
            generator_optimizer.step()
            
            # Print loss and save model
            if i % log_interval == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(dataloader)}] Discriminator Loss: {real_loss + fake_loss:.4f} Generator Loss: {generator_loss:.4f}")
                torch.save({
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'generator_optimizer_state_dict': generator_optimizer.state_dict(),
                    'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict(),
                }, f'saved_models/epoch_{epoch}_/cgan_model_{i}.pth')

        # Generate images for each class
        for i in range(num_classes):
            # Generate fake images for the current class
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_labels = torch.zeros(batch_size, num_classes, device=device)
            fake_labels[:, i] = 1
            fake_images = generator(noise, fake_labels).detach().cpu()

            # Rescale the images to [0, 1]
            fake_images = (fake_images + 1) / 2

            # Save the images as a grid
            save_image(fake_images, f"images/epoch_{epoch + 1}_/class_{i}.png", nrow=int(math.sqrt(batch_size)))


# Parameters need to initialize generator and discriminator
NUM_CLASSES = 12
LATENT_DIM = 64
IMAGE_SHAPE = (3, 108, 108)
EPOCH = 400
LOG_INTERVAL = 100

generator = CGANGenerator(64, 12, (3, 108, 108))
discriminator = CGANDiscriminator(12)

train_cgan(generator, discriminator, dataloader, device, NUM_CLASSES, LATENT_DIM, EPOCH, LOG_INTERVAL)