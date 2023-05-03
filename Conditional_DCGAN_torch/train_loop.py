import torch

def train_step(generator, 
               discriminator, 
               g_opt, 
               d_opt, 
               g_loss, 
               d_loss, 
               images, 
               labels, 
               device, 
               latent_space_dim):
    
    images.to(device)
    labels.to(device)
    labels = labels.unsqueeze(1)  # (batch) -> (batch,1) in order to be fed to the model

    real_targets = torch.ones((images.size(0), 1)).to(device)
    fake_targets = torch.zeros((images.size(0), 1)).to(device)

    real_outputs = discriminator((images, labels))
    d_real_loss = d_loss(real_outputs, real_targets)

    noise_vector = torch.randn(images.size(0), latent_space_dim, device=device)
    fake_images = generator((noise_vector, images))
    discriminator_fake_outputs = discriminator((fake_images.detach(), labels))
    

    return