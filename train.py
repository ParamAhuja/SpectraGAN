# CELL 5: THE SPECTRAGAN TRAINING LOOP
import torch
import torch.nn as nn
import torch.optim as optim
import os

# 1. SETUP HARDWARE AND MODELS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# Initialize models (from Cell 1 and Cell 2)
generator = SpectraGAN_Generator().to(device)
discriminator = SpectraGAN_Discriminator().to(device)

# Initialize Loss (from Cell 3)
criterion = SpectraGAN_Loss().to(device)
bce_loss = nn.BCEWithLogitsLoss().to(device) # For the Adversarial RaGAN part

# Optimizers (Standard ESRGAN parameters)
optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.9, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.9, 0.999))

# Create folder to save weights
os.makedirs("/kaggle/working/weights", exist_ok=True)

# 2. THE TRAINING LOOP
epochs = 5 # Set to 5 just to test the loop tonight. For real training, use 50-100.
print("Starting SpectraGAN Training...")

for epoch in range(epochs):
    generator.train()
    discriminator.train()
    
    running_g_loss = 0.0
    running_d_loss = 0.0
    
    # train_loader comes from Cell 4
    for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
        # Move tensors to GPU
        lr_imgs = lr_imgs.to(device)
        real_hr = hr_imgs.to(device)
        
        # Labels for RaGAN
        real_label = torch.ones((lr_imgs.size(0), 1)).to(device)
        fake_label = torch.zeros((lr_imgs.size(0), 1)).to(device)
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        
        # Generate fake High-Res images
        fake_hr = generator(lr_imgs)
        
        # Predict realism
        pred_real = discriminator(real_hr)
        pred_fake = discriminator(fake_hr.detach()) # Detach so we don't backprop through G
        
        # Relativistic Average Loss (RaGAN)
        loss_d_real = bce_loss(pred_real - pred_fake.mean(), real_label)
        loss_d_fake = bce_loss(pred_fake - pred_real.mean(), fake_label)
        loss_D = (loss_d_real + loss_d_fake) / 2
        
        loss_D.backward()
        optimizer_D.step()
        
        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        
        # Predict realism again (for Generator update)
        pred_fake_g = discriminator(fake_hr)
        pred_real_g = discriminator(real_hr.detach())
        
        # Generator Adversarial Loss (Generator wants fake to be scored as real)
        g_adv_loss_fake = bce_loss(pred_fake_g - pred_real_g.mean(), real_label)
        g_adv_loss_real = bce_loss(pred_real_g - pred_fake_g.mean(), fake_label)
        g_adv_loss = (g_adv_loss_fake + g_adv_loss_real) / 2
        
        # SpectraGAN Multi-Spectral Loss (L1 + VGG)
        g_content_loss, pix_loss, perc_loss = criterion(fake_hr, real_hr)
        
        # Combine losses (Standard weighting: Content loss is heavily prioritized early on)
        loss_G = g_content_loss + (0.005 * g_adv_loss)
        
        loss_G.backward()
        optimizer_G.step()
        
        # Tracking
        running_d_loss += loss_D.item()
        running_g_loss += loss_G.item()
        
        if i % 10 == 0:
            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(train_loader)}] "
                  f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}] "
                  f"(Pix: {pix_loss.item():.4f}, Perc: {perc_loss.item():.4f})")

    # Save model weights every epoch
    torch.save(generator.state_dict(), f"/kaggle/working/weights/SpectraGAN_Gen_Epoch_{epoch}.pth")
    print(f"==== Epoch {epoch} Complete. Model Saved. ====\n")

print("TRAINING SUCCESSFUL! Objective 2 is Complete.")