import torch
from torch import nn


def custom_weights_init(model):
    """
    the article uses particular initializations for the weights
    of convs and normalization layers
    """
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, latent_space_dim, img_size=28, img_channels=1, n_classes=10, embedding_dim=50):
        super().__init__()
        self.latent_space_dim = latent_space_dim
        self.img_size = img_size
        self.img_channels = img_channels
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim

        self.latent = nn.Sequential(
            # to be reshaped as (batch_size, img_size*4, 7, 7) in the forward()
            nn.Linear(self.latent_space_dim, 7*7*self.img_size*4),
            nn.LeakyReLU(0.2)
        )

        self.label_embedding = nn.Sequential(
            nn.Embedding(self.n_classes, self.embedding_dim),
            nn.Linear(self.embedding_dim, 7*7)
        )

        self.main = nn.Sequential(
            # in_channels must have a +1 for the label embedding
            nn.ConvTranspose2d(self.img_size*4+1, self.img_size*2, kernel_size=4, stride=1, padding=1, bias=False),  # (batch,28*2,14,14)
            nn.BatchNorm2d(self.img_size*2),
            nn.ReLU(),
            nn.ConvTranspose2d(self.img_size*2, self.img_size, kernel_size=4, stride=1, padding=1, bias=False),  # (batch,28,28,28)
            nn.BatchNorm2d(self.img_size),
            nn.ReLU(),
            nn.Conv2d(self.img_size, self.img_channels, kernel_size=3, stride=1, padding=1, bias=False),  # (batch,img_channels,28,28)
            nn.Tanh()  # in the article pixel values are normalized [-1,1]
        )


    def forward(self, input):
        
        noise_vector, label = input

        latent_noise = self.latent(noise_vector)
        latent_noise = latent_noise.view(-1, self.img_size*4, 7, 7)

        embedded_label = self.label_embedding(label)
        embedded_label = embedded_label.view(-1, 1, 7, 7)

        inputs = torch.cat((latent_noise, embedded_label), dim=1)
        generated_image = self.main(inputs)

        return generated_image 
    

class Discriminator(nn.Module):
    def __init__(self, img_size=28, img_channels=1, n_classes=10, embedding_dim=50):
        super().__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim

        self.label_embedding = nn.Sequential(
            nn.Embedding(self.n_classes, self.embedding_dim),
            nn.Linear(self.embedding_dim, self.img_size**2)
        )

        self.main = nn.Sequential(
            # same +1 in in_channels for the label embedding
            nn.Conv2d(self.img_channels+1, self.img_size, kernel_size=4, stride=2, padding=1, bias=False),  # (batch,28,14,14)
            nn.LeakyReLU(0.2), # the article uses leaky ReLU with slope 0.2
            nn.Conv2d(self.img_size, self.img_size*2, kernel_size=4, stride=2, padding=1, bias=False),  # (batch,28*2,7,7)
            nn.BatchNorm2d(self.img_size*2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.img_size*2, self.img_size*4, kernel_size=3, stride=1, padding=1, bias=False),  # (batch,28*4,7,7)
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(28*4*7*7,1),
            nn.Sigmoid()
        )


    def forward(self, input):
        
        image, label = input

        embedded_label = self.label_embedding(label)
        embedded_label = embedded_label.view(-1,1,self.img_size,self.img_size)

        inputs = torch.cat((image, embedded_label), sim=1)
        result = self.main(inputs)
        
        return result