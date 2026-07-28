import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset, load_from_disk
import numpy as np
import tifffile
import io

class HFHySpecNetDataset(Dataset):
    def __init__(self, split='train', transform=None):
        # Loads directly via HF API to bypass local torchgeo extraction corruption
        drive_cache = "/content/drive/MyDrive/hyspecnet11k/datasets"
        self.data_path = ""
        self.swiper = True
        
        if self.swiper:
            self.dataset = load_dataset("torchgeo/hyspecnet", cache_dir=drive_cache, split="train")
        else:
            self.dataset = load_from_disk(self.data_path)
            
        # cast_column("image", Image(decode=False)) removed because we bypass HF formatting entirely
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        # BYPASS Hugging Face's formatter to avoid PIL UnidentifiedImageError.
        # Access the underlying Apache Arrow table to extract the raw bytes directly.
        img_dict = self.dataset.data.column('image')[idx].as_py()
        img_bytes = img_dict['bytes']
        
        # Decode the raw bytes safely using tifffile
        img = tifffile.imread(io.BytesIO(img_bytes))
        
        # Convert the array to a float32 PyTorch tensor
        img = np.array(img, dtype=np.float32)
        img = torch.from_numpy(img)
        
        # I-JEPA expects channel-first format (224, 128, 128)
        # If the HF dataset yields channels-last (128, 128, 224), swap them:
        if img.shape[-1] == 224:
            img = img.permute(2, 0, 1)
            
        if self.transform:
            img = self.transform(img)
            
        # I-JEPA SSL ignores labels, so we return 0 as a dummy target
        return img, 0 

def make_hyspecnet_loader(transform, batch_size, collator, num_workers=0):
    dataset = HFHySpecNetDataset(split='train', transform=transform)
    sampler = DistributedSampler(dataset)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers, # Defaulting to 0 to prevent FUSE filesystem crashes
        pin_memory=True,
        sampler=sampler,
        drop_last=True,
        collate_fn=collator
    )
    return dataloader, sampler
