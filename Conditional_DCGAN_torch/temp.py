import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

# hyperparameters
batch_size = 32
epochs = 20
latent_space_dim = 50


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5,0.5)
])

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

images, labels = next(iter(train_dataloader))

print(labels.unsqueeze(1).dtype)

