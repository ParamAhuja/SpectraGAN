# CELL 6: SPECTRAGAN EVALUATION AND BENCHMARKING
import torch
import numpy as np
import os
import cv2
from torch.utils.data import DataLoader
from datasets import load_dataset
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

# --- 1. SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluating on: {device}")

# We need the classes you defined in Cell 1 and Cell 4
# (Make sure SpectraGAN_Generator and SpectraGANDataset are loaded in memory)
model = SpectraGAN_Generator(in_nc=4, out_nc=4).to(device)

# Load your best trained weights (Update the epoch number to your best run)
WEIGHTS_PATH = "/kaggle/working/weights/SpectraGAN_Gen_Epoch_4.pth" 
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval() # CRITICAL: Turns off dropout and batchnorm updates

# --- 2. LOAD TEST DATA ---
# Replace with your actual Hugging Face Repo Name
REPO_NAME = "your-username/SpectraGAN-SEN12MS-PartA"
print(f"Loading Test Split from {REPO_NAME}...")

hf_test_data = load_dataset(REPO_NAME, split="test")
test_dataset = SpectraGANDataset(hf_test_data)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# --- 3. METRICS TRACKING ---
total_psnr = 0.0
total_ssim = 0.0
num_samples = len(test_loader)

# Create a folder to save visual results for your thesis
OUTPUT_DIR = "/kaggle/working/eval_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Starting Evaluation on {num_samples} unseen images...")

# --- 4. THE EVALUATION LOOP ---
with torch.no_grad(): # Disables gradient calculation to save memory/speed
    for i, (lr_imgs, hr_imgs) in enumerate(tqdm(test_loader, desc="Benchmarking")):
        lr_imgs = lr_imgs.to(device)
        real_hr = hr_imgs.to(device)
        
        # Generate the Super-Resolution image
        fake_hr = model(lr_imgs)
        
        # Clamp values to valid image range [0.0, 1.0] to prevent metric explosion
        fake_hr = torch.clamp(fake_hr, 0.0, 1.0)
        
        # Convert PyTorch Tensors [Batch, Channels, Height, Width] 
        # to Numpy Arrays [Height, Width, Channels] for scikit-image
        fake_np = fake_hr.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        real_np = real_hr.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        
        # --- CALCULATE METRICS (Across all 4 RGB+NIR Channels) ---
        # PSNR: Higher is better (less noise)
        batch_psnr = peak_signal_noise_ratio(real_np, fake_np, data_range=1.0)
        
        # SSIM: Closer to 1.0 is better (perfect structural match)
        batch_ssim = structural_similarity(real_np, fake_np, data_range=1.0, channel_axis=-1)
        
        total_psnr += batch_psnr
        total_ssim += batch_ssim
        
        # --- SAVE VISUAL PROOF (First 10 images only) ---
        if i < 10:
            # We must strip the 4th (NIR) channel because standard monitors/PNGs only show RGB
            # Slicing [:, :, :3] takes only Red, Green, and Blue.
            # OpenCV expects BGR, so we flip RGB to BGR using [:, :, ::-1]
            viz_fake_bgr = (fake_np[:, :, :3][:, :, ::-1] * 255).astype(np.uint8)
            viz_real_bgr = (real_np[:, :, :3][:, :, ::-1] * 255).astype(np.uint8)
            
            # Also upscale the LR image using standard Bicubic for a baseline comparison
            lr_np = lr_imgs.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            viz_lr_bgr = (lr_np[:, :, :3][:, :, ::-1] * 255).astype(np.uint8)
            viz_bicubic_bgr = cv2.resize(viz_lr_bgr, (256, 256), interpolation=cv2.INTER_CUBIC)
            
            # Stitch them side-by-side: [Bicubic | SpectraGAN | Ground Truth]
            comparison = np.hstack((viz_bicubic_bgr, viz_fake_bgr, viz_real_bgr))
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"comparison_{i}_psnr{batch_psnr:.2f}.png"), comparison)

# --- 5. FINAL RESULTS ---
avg_psnr = total_psnr / num_samples
avg_ssim = total_ssim / num_samples

print("\n" + "="*40)
print(" SPECTRA-GAN BENCHMARKING RESULTS")
print("="*40)
print(f" Total Test Images Evaluated: {num_samples}")
print(f" Average PSNR: {avg_psnr:.4f} dB")
print(f" Average SSIM: {avg_ssim:.4f}")
print("="*40)
print(f"Saved 10 comparison images to {OUTPUT_DIR} for your thesis document!")