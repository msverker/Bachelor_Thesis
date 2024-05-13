import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import os

class BhutanDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Initialize the dataset by listing all .pt files in the data directory.
        Args:
            data_dir (str): Path to the directory containing the .pt files.
            transform (callable, optional): Optional transform to apply to the spectrogram tensors.
        """
        self.data_dir = Path(data_dir)
        self.file_paths = list(self.data_dir.glob('*.pt'))
        self.transform = transform

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.file_paths)

    def __getitem__(self, index):
        """
        Retrieve the tensor and label at the specified index.
        Args:
            index (int): Index of the data to retrieve.
        
        Returns:
            tuple: (tensor, label) where tensor is the resized spectrogram tensor and
                   label is the corresponding label.
        """
        # Load data from a .pt file
        data = torch.load(self.file_paths[index])
        spectrogram = data['spectrogram']
        label = int(data['label'])

        # Apply the transformation (resize in this case)
        if self.transform:
            spectrogram = self.transform(spectrogram)

        return spectrogram, label

# Define the transformation: Resize the images to 224x224
transform = transforms.Compose([
    transforms.Resize((224, 224))
])

# Assuming the data is stored in '/path/to/your/data'
data_dir = '/zhome/09/8/169747/Bachelor_Thesis/Bhutan_spectrograms'

# Create an instance of the dataset
Bhutan_dataset = BhutanDataset(data_dir, transform=transform)