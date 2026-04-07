import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from plotting_utils import plot_vae_loss, plot_vae_latent_dim_comparison

# Define Variational Autoencoder (VAE)
class VAE(nn.Module):
    def __init__(self, hidden_dim=200, latent_dim=20):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28,hidden_dim),
            nn.ReLU()
        ) #encoder
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim) #mu vector
        self.fc_sigma = nn.Linear(hidden_dim, latent_dim) #sigma vector
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 28*28),
            nn.Sigmoid()
        ) #decoder

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        
        # To prevent sigma from being negative, which isn't a valid std dev
        logvar = self.fc_sigma(h)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma

    def reparameterize(self, mu, sigma):
        z = mu + sigma * torch.randn_like(sigma)
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, sigma = self.encode(x.view(-1, 28*28))
        z = self.reparameterize(mu, sigma)
        return self.decode(z), mu, sigma
        
def loss_function(criterion,recon_x, x, mu, sigma, beta):
    target = x.view(-1, 28*28)

    if isinstance(criterion, nn.BCELoss):
        RECON = nn.functional.binary_cross_entropy(recon_x, target, reduction="sum")
    elif isinstance(criterion, nn.MSELoss):
        RECON = nn.functional.mse_loss(recon_x, target, reduction="sum")
    else:
        RECON = criterion(recon_x, target)
    
    KLD = -0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
    return RECON + beta * KLD, RECON.item(), KLD.item()

# Inference
def generate_images(vae, latent_dim, device, num_images=10, filename="unconditional_generation.png"):
    vae.eval()
    z = torch.randn(num_images, latent_dim).to(device)
    with torch.no_grad():
        generated_imgs = vae.decode(z).cpu()
    plt.figure(figsize=(num_images * 1.5, 2))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(generated_imgs[i].view(28, 28), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"vae_results/{filename}")
    plt.close()

def find_mnist_image_target(dataloader, target_digit):
    for imgs, labels in dataloader:
        matches = (labels == target_digit).nonzero(as_tuple=True)[0]
        if len(matches) > 0:
            return imgs[matches[0]].unsqueeze(0)
    return None

def conditional_generate_images(vae, latent_dim, target_digit, dataloader, device="cpu", num_images=10, filename=None):
    vae.eval()
    target_image = find_mnist_image_target(dataloader, target_digit)
    if target_image is None:
        raise ValueError(f"No image found for digit {target_digit}")

    eps = torch.randn(num_images, latent_dim).to(device)
    
    with torch.no_grad():
        mu, sigma = vae.encode(target_image.view(-1, 28*28).to(device))
        z = mu + sigma * eps
        # z = vae.reparameterize(mu, sigma)
        generated_imgs = vae.decode(z).cpu()

    if filename is None:
        filename = f"conditional_generation_digit_{target_digit}.png"

    plt.figure(figsize=(num_images * 1.5, 2))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(generated_imgs[i].view(28, 28), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"vae_results/{filename}")
    plt.close()

def train(latent_dim=20, device="cpu", criterion=nn.BCELoss(), epochs=10, beta=1):
    vae = VAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=3e-4)

    dataloader = DataLoader(datasets.MNIST('.', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])), batch_size=32, shuffle=True)

    # Training Loop
    plot_data = []
    for epoch in range(epochs):
        vae.train()
        total_loss = 0.0
        total_reconstruction_loss = 0.0
        total_kld_loss = 0.0
        num_imgs = len(dataloader.dataset)
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            gen_imgs = vae(imgs)
            
            loss, reconstruction_loss, kld_loss = loss_function(criterion, gen_imgs[0], imgs, gen_imgs[1], gen_imgs[2], beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_reconstruction_loss += reconstruction_loss
            total_kld_loss += kld_loss

        loss_info = {"loss": total_loss/num_imgs, "reconstruction_loss": total_reconstruction_loss/num_imgs, "kld_loss": total_kld_loss/num_imgs}
        plot_data.append(loss_info)

        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss/len(dataloader.dataset):.4f}")

    criterion_name = criterion.__class__.__name__.lower().replace("loss", "")
    run_label = f"{criterion_name}_latent_{latent_dim}_beta_{beta}"

    # Plot training loss
    plot_vae_loss(plot_data, filename=f"vae_loss_{run_label}.png")

    # Unconditional generation
    generate_images(vae, latent_dim, device, num_images=10, filename=f"unconditional_generation_{run_label}.png")
    
    target_digit = 7
    conditional_generate_images(
        vae,
        latent_dim,
        target_digit,
        dataloader,
        device=device,
        num_images=10,
        filename=f"conditional_generation_digit_{target_digit}_{run_label}.png",
    )

    return vae, plot_data
    
    

def main():
    os.makedirs("vae_results", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    latent_dim = 20
    epochs = 100
    
    # Default training with BCE loss
    _, bce_plot_data = train(latent_dim=latent_dim, device=device, criterion=nn.BCELoss(), epochs=epochs)

    # Training with MSE loss instead of BCE
    train(latent_dim=latent_dim, device=device, criterion=nn.MSELoss(), epochs=epochs)

    latent_dim_histories = [(20, bce_plot_data)]
    latent_dims = [10,50]
    for latent_dim in latent_dims:
        print("Training with latent dimension of", latent_dim)
        _, plot_data = train(latent_dim=latent_dim, device=device, epochs=epochs)
        latent_dim_histories.append((latent_dim, plot_data))

    plot_vae_latent_dim_comparison(latent_dim_histories)

if __name__ == "__main__":
    main()
