# CELL 3: SPECTRAGAN MULTI-SPECTRAL LOSS FUNCTION
import torch
import torch.nn as nn
import torchvision.models as models

class VGGFeatureExtractor(nn.Module):
    """
    Extracts high-level texture features from VGG19.
    Required for ESRGAN's perceptual loss.
    """
    def __init__(self, feature_layer=34, use_bn=False):
        super(VGGFeatureExtractor, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        # We extract features before the activation of the target layer
        self.features = nn.Sequential(*list(vgg19.features.children())[:feature_layer + 1])
        
        # Freeze VGG weights (we are only using it as a judge, not training it)
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        return self.features(x)

class SpectraGAN_Loss(nn.Module):
    """
    Custom Loss Function for 4-Channel Earth Observation Data.
    Combines strict L1 Pixel Loss (RGB+NIR) with VGG Perceptual Loss (RGB only).
    """
    def __init__(self):
        super(SpectraGAN_Loss, self).__init__()
        self.vgg = VGGFeatureExtractor()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss() # Used for VGG feature comparison

    def forward(self, generated_hr, real_hr):
        # 1. PIXEL LOSS (Radiometric Accuracy)
        # Calculate L1 loss across ALL 4 CHANNELS (Red, Green, Blue, NIR)
        pixel_loss = self.l1_loss(generated_hr, real_hr)

        # 2. PERCEPTUAL TEXTURE LOSS (Structural Realism)
        # TENSOR SLICING: Extract only the first 3 channels (RGB) for the VGG network
        gen_rgb = generated_hr[:, :3, :, :]
        real_rgb = real_hr[:, :3, :, :]
        
        # Extract features
        gen_features = self.vgg(gen_rgb)
        real_features = self.vgg(real_rgb).detach() # Detach real features from graph
        
        # Compare textures
        perceptual_loss = self.mse_loss(gen_features, real_features)

        # Combine losses (Standard ESRGAN weighting: 1.0 for Perceptual, 0.01 for Pixel)
        # Note: You can adjust these weights later to favor structure vs color
        total_generator_loss = perceptual_loss + (0.01 * pixel_loss)
        
        return total_generator_loss, pixel_loss, perceptual_loss

# --- TEST THE LOSS FUNCTION ---
if __name__ == "__main__":
    # Simulate a batch of 2 generated vs real 4-channel 256x256 images
    dummy_generated = torch.randn(2, 4, 256, 256)
    dummy_real = torch.randn(2, 4, 256, 256)
    
    loss_fn = SpectraGAN_Loss()
    total_loss, pix_loss, perc_loss = loss_fn(dummy_generated, dummy_real)
    
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Pixel Loss (4-Channel): {pix_loss.item():.4f}")
    print(f"Perceptual Loss (3-Channel Sliced): {perc_loss.item():.4f}")
    print("SUCCESS: Multi-Spectral Loss function compiled and sliced tensors correctly!")