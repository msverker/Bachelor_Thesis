import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import torchvision
import seaborn as sns
from random import sample
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class SpectrogramDataset(Dataset):
    def __init__(self, spectro_folder):
        self.spectrogram_folders = list(spectro_folder.glob('spectrogram_*'))
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.spectrogram_folders)

    def __getitem__(self, idx):
        folder_path = self.spectrogram_folders[idx]

        # Load the spectrogram image
        img = Image.open(folder_path / 'spectrogram.png').convert('RGB')
        img = self.transform(img)

        # Load the label
        label_path = folder_path / 'label.txt'
        with open(label_path, 'r') as label_file:
            label = int(label_file.read().strip()) - 1

        return img, label

# Path to the folder containing your spectrograms
spectro_folder = Path('./spectrograms')
#If on HPC
# spectro_folder = Path('')

# Create a custom dataset
Data = SpectrogramDataset(spectro_folder)

#For SVM loader
unique_labels = set()

for _, label in Data:
    unique_labels.add(label)

class_indices = {label: [] for label in unique_labels}

for idx in range(len(Data)):
    _, label = Data[idx]
    class_indices[label].append(idx)

selected_indices = []
for label, indices in class_indices.items():
    if len(indices) >= 100:
        selected_indices.extend(sample(indices, 100))
    else:
        selected_indices.extend(indices)
        # print(f"Not enough data in class {label}, only {len(indices)} available.")

subset_dataset = Subset(Data, selected_indices)

subset_dataloader = DataLoader(subset_dataset, batch_size=len(subset_dataset), shuffle=True)  # Adjust batch size as needed

images, labels = next(iter(subset_dataloader))
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels, random_state=42)

train_idx, test_idx = train_test_split(selected_indices, test_size=0.2, stratify=labels, random_state=42)
all_indices = set(range(len(Data)))
remaining_indices = list(all_indices - set(test_idx))
remaining_dataset = Subset(Data, remaining_indices)
