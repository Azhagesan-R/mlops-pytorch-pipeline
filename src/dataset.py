import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class SyntheticImageDataset(Dataset):
    """Generates synthetic image tensors to ensure zero-dependency pipeline execution."""
    def __init__(self, num_samples: int = 100, num_classes: int = 10):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.transform = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        # Generate a synthetic 3-channel 32x32 image image
        x = torch.rand(3, 32, 32)
        x = self.transform(x)
        y = torch.randint(0, self.num_classes, (1,)).item()
        return x, y

def get_dataloader(batch_size: int, num_samples: int) -> DataLoader:
    dataset = SyntheticImageDataset(num_samples=num_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
