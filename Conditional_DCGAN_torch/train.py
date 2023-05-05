import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import train_loop
import models
from matplotlib import pyplot as plt
import time

# hyperparameters
batch_size = 64
epochs = 5
latent_space_dim = 100
lr = 0.0002
beta1 = 0.5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5,0.5)
])

training_data = datasets.MNIST(
    root="data_mnist",
    train=True,
    download=True,
    transform=transform
)


train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

discriminator = models.Discriminator()
generator = models.Generator(latent_space_dim)
d_opt = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
g_opt = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
criterion = torch.nn.BCELoss()

d_losses = []
g_losses = []

for epoch in range(epochs):
    start = time.time()
    for iter, (images, labels) in enumerate(train_dataloader):
        d_err, g_err = train_loop.train_step(discriminator=discriminator,
                                            generator=generator,
                                            d_opt=d_opt,
                                            g_opt=g_opt,
                                            d_loss=criterion,
                                            g_loss=criterion,
                                            images=images,
                                            labels=labels,
                                            device=device,
                                            latent_space_dim=latent_space_dim)
        d_losses.append(d_err.item())
        g_losses.append(g_err.item())
    end = time.time()
    elapsed = end-start
    print("Epoch %d done, time: %.1fs" %(epoch, elapsed))
    
plt.plot(d_losses, label='d_loss')
plt.plot(g_losses, label='g_loss')
plt.legend()


noise_vector = torch.randn(10, latent_space_dim, device=device)
fake_labels = torch.randint(0, 9, (10,))
fake_images = generator((noise_vector, fake_labels))

fig, axs = plt.subplots(2, 5)
plt.suptitle('Fake images')
for i in range(10):
    row = i // 5
    col = i % 5
    image_np = fake_images[i].permute(1, 2, 0).detach().numpy()
    axs[row, col].imshow(image_np, cmap='gray')
    axs[row, col].set_title(fake_labels[i].numpy())
    
plt.show()