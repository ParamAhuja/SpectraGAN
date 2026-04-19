# CELL 2: THE SPECTRAGAN DISCRIMINATOR (VGG-Style)
import torch
import torch.nn as nn

class SpectraGAN_Discriminator(nn.Module):
    """
    VGG-style Relativistic Discriminator modified for Earth Observation.
    Evaluates 4-channel (RGB+NIR) 256x256 images.
    """
    def __init__(self, in_nc=4, base_nf=64):
        super(SpectraGAN_Discriminator, self).__init__()
        
        # 1. Multi-Spectral Input Layer
        self.conv0_0 = nn.Conv2d(in_nc, base_nf, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(base_nf, base_nf, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(base_nf, affine=True)

        # 2. Deep Feature Extraction (Progressively downsampling 256 -> 128 -> 64 -> 32 -> 16 -> 8)
        self.conv1_0 = nn.Conv2d(base_nf, base_nf * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(base_nf * 2, affine=True)
        self.conv1_1 = nn.Conv2d(base_nf * 2, base_nf * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(base_nf * 2, affine=True)

        self.conv2_0 = nn.Conv2d(base_nf * 2, base_nf * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(base_nf * 4, affine=True)
        self.conv2_1 = nn.Conv2d(base_nf * 4, base_nf * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(base_nf * 4, affine=True)

        self.conv3_0 = nn.Conv2d(base_nf * 4, base_nf * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(base_nf * 8, affine=True)
        self.conv3_1 = nn.Conv2d(base_nf * 8, base_nf * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(base_nf * 8, affine=True)

        self.conv4_0 = nn.Conv2d(base_nf * 8, base_nf * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(base_nf * 8, affine=True)
        self.conv4_1 = nn.Conv2d(base_nf * 8, base_nf * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(base_nf * 8, affine=True)

        # 3. Dense Classifier (Outputs the raw logit for "Realness")
        self.linear1 = nn.Linear(base_nf * 8 * 8 * 8, 100)
        self.linear2 = nn.Linear(100, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.lrelu(self.conv0_0(x))
        x = self.lrelu(self.bn0_1(self.conv0_1(x)))
        
        x = self.lrelu(self.bn1_0(self.conv1_0(x)))
        x = self.lrelu(self.bn1_1(self.conv1_1(x)))
        
        x = self.lrelu(self.bn2_0(self.conv2_0(x)))
        x = self.lrelu(self.bn2_1(self.conv2_1(x)))
        
        x = self.lrelu(self.bn3_0(self.conv3_0(x)))
        x = self.lrelu(self.bn3_1(self.conv3_1(x)))
        
        x = self.lrelu(self.bn4_0(self.conv4_0(x)))
        x = self.lrelu(self.bn4_1(self.conv4_1(x)))

        x = x.view(x.size(0), -1) # Flatten the tensor
        x = self.lrelu(self.linear1(x))
        out = self.linear2(x)
        return out

# --- TEST THE ARCHITECTURE ---
if __name__ == "__main__":
    # Simulate a generated batch of 2 High-Res (256x256) 4-channel images
    dummy_hr_input = torch.randn(2, 4, 256, 256)
    discriminator = SpectraGAN_Discriminator()
    
    score = discriminator(dummy_hr_input)
    print(f"Input Shape: {dummy_hr_input.shape}")
    print(f"Discriminator Output Shape: {score.shape}")
    print("SUCCESS: Discriminator fully operational and reducing to logit scores!")