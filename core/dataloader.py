# CELL 4: SPECTRAGAN HUGGING FACE DATALOADER
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torchvision.transforms as transforms

from huggingface_hub import login
from kaggle_secrets import UserSecretsClient



class SpectraGANDataset(Dataset):
    """
    Custom PyTorch Dataset wrapper for Hugging Face Earth Observation data.
    Ensures 4-channel (RGB+NIR) PIL images are correctly transformed into PyTorch Tensors.
    """
    def __init__(self, hf_dataset_split):
        self.dataset = hf_dataset_split
        # ToTensor() automatically converts PIL images to [C, H, W] tensors
        # and scales pixel values from 0-255 down to 0.0-1.0
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Fetch the paired images from Hugging Face
        lr_pil = self.dataset[idx]['lr']
        hr_pil = self.dataset[idx]['hr']
        
        # Ensure they are read as RGBA (4-channel) to preserve the NIR band
        if lr_pil.mode != 'RGBA':
            lr_pil = lr_pil.convert('RGBA')
        if hr_pil.mode != 'RGBA':
            hr_pil = hr_pil.convert('RGBA')
            
        # Convert to PyTorch Tensors
        lr_tensor = self.transform(lr_pil)
        hr_tensor = self.transform(hr_pil)
        
        return lr_tensor, hr_tensor

# --- TEST THE DATALOADER ---
if __name__ == "__main__":
    print("Connecting to Hugging Face...")
    user_secrets = UserSecretsClient()
    secret_value_0 = user_secrets.get_secret("HF_TOKEN")
    login(token = secret_value_0)

    # IMPORTANT: Replace this with the exact name of the dataset you pushed earlier!
    REPO_NAME = "ParamDev/SpectraGAN-SEN12MS_AllSeasons_1k" 
    
    try:
        # Load just the 'train' split
        hf_train_data = load_dataset(REPO_NAME, split="train")
        
        # Wrap it in our custom PyTorch Dataset
        train_dataset = SpectraGANDataset(hf_train_data)
        
        # Load it into a PyTorch DataLoader (Batch size 8 is safe for Kaggle GPUs)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
        
        # Fetch a single batch to test the pipeline
        lr_batch, hr_batch = next(iter(train_loader))
        
        print(f"Success! Fetched a batch of {len(lr_batch)} images.")
        print(f"Low-Res Input Tensor Shape: {lr_batch.shape} (Expected: [8, 4, 64, 64])")
        print(f"High-Res Target Tensor Shape: {hr_batch.shape} (Expected: [8, 4, 256, 256])")
        
    except Exception as e:
        print(f"Could not load dataset. Ensure your REPO_NAME is correct and your Kaggle Internet is ON.\nError: {e}")