import torch

def train_step(discriminator, 
               generator, 
               d_opt, 
               g_opt, 
               d_loss, 
               g_loss, 
               images, 
               labels, 
               device, 
               latent_space_dim):
    
    images.to(device)
    labels.to(device)
    labels = labels.unsqueeze(1)  # (batch) -> (batch,1) in order to be fed to the model

    real_targets = torch.ones((images.size(0), 1)).to(device)
    fake_targets = torch.zeros((images.size(0), 1)).to(device)

    # discriminator
    d_opt.zero_grad()

    real_outputs = discriminator((images, labels))
    d_real_loss = d_loss(real_outputs, real_targets)

    noise_vector = torch.randn(images.size(0), latent_space_dim, device=device)
    fake_images = generator((noise_vector, labels))
    discriminator_fake_outputs = discriminator((fake_images.detach(), labels))
    d_fake_loss = d_loss(discriminator_fake_outputs,fake_targets)

    d_total_loss = (d_real_loss + d_fake_loss)/2

    d_total_loss.backward()
    d_opt.step()

    # generator
    g_opt.zero_grad()
    g_total_loss = g_loss(discriminator((fake_images, labels)), real_targets)
    g_total_loss.backward()
    g_opt.step()


    return d_total_loss, g_total_loss

