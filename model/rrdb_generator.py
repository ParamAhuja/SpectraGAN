# CELL 1: THE SPECTRAGAN GENERATOR (RRDBNet)
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDenseBlock_5C(nn.Module):
    """Standard Dense Block with 5 Convolutional Layers"""
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    """Residual in Residual Dense Block"""
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class SpectraGAN_Generator(nn.Module):
    """
    Modified RRDBNet for Earth Observation.
    in_nc: 4 (RGB + NIR)
    out_nc: 4 (RGB + NIR)
    """
    def __init__(self, in_nc=4, out_nc=4, nf=64, nb=23, gc=32):
        super(SpectraGAN_Generator, self).__init__()
        RRDB_block_f = RRDB
        
        # 1. Multi-Spectral Input Layer (The crucial 4-channel modification)
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        
        # 2. Deep Feature Extraction (The RRDB Trunk)
        self.RRDB_trunk = nn.Sequential(*[RRDB_block_f(nf, gc) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        # 3. Upsampling (64x64 -> 128x128 -> 256x256)
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        # 4. High-Resolution Output Layers (Modified for 4 channels)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(feat))
        feat = feat + trunk
        
        # Upsampling (Nearest neighbor + Conv prevents checkerboard artifacts)
        feat = self.lrelu(self.upconv1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.upconv2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        
        out = self.conv_last(self.lrelu(self.HRconv(feat)))
        return out

# --- TEST THE ARCHITECTURE ---
if __name__ == "__main__":
    # Simulate a batch of 2 images, 4 channels (RGB+NIR), 64x64 resolution
    dummy_input = torch.randn(2, 4, 64, 64)
    model = SpectraGAN_Generator()
    
    output = model(dummy_input)
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
    print("SUCCESS: SpectraGAN Generator successfully compiled and processed 4-channel data!")