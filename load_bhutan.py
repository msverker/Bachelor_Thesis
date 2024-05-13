import cv2
import io
from skimage.util import img_as_float

def plot_to_tensor(time_grid, freq_grid, power, times, frequencies):
    # Create a figure and an axes object
    fig, ax = plt.subplots()
    # Plot using pcolor with a logarithmic color normalization
    c = ax.pcolor(time_grid, freq_grid, power, norm=LogNorm(), cmap='gray')
    ax.set_xlim(times.min(), times.max())
    ax.set_ylim(frequencies.min(), frequencies.max())
    # Remove axes for cleaner image
    ax.axis('off')
    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    fig.clf()
    buf.seek(0)
    # Open the image and convert to grayscale
    img = Image.open(buf).convert('L')
    # Convert image to a numpy array
    img_arr = np.array(img)
    img_tensor = torch.tensor(img_arr, dtype=torch.float32)
    return img_tensor

import h5py
from pathlib import Path
import numpy as np
import json
from cwt_spectrogram import *
import os
from PIL import Image
import torch

DPI = 300
output_dir = Path('/zhome/09/8/169747/Bachelor_Thesis/Bhutan_spectrograms')
file_path = Path("/dtu-compute/EEG_at_scale/preprocess/preprocess_downstream_bhutan_noica_5.0_combined/combined_00000.hdf5")
sampling_freq = 256

with h5py.File(file_path, 'r') as file:
    attribute_manager = file.attrs['descriptions']
    print(attribute_manager)
    num_events = file['data'].shape[0]
    for event_index in range(num_events):
        output_file_path = os.path.join(output_dir, f'event_{event_index}.pt')

        if os.path.exists(output_file_path):
            print(f'File for event {event_index} already exists, skipping...')
            continue
        
        event_data = file['data'][event_index]
        spectrogram_tensors = []
        for channel in event_data:
            power, times, frequencies, _ = cwt_spectrogram(channel, sampling_freq, nNotes=4)
            time_grid, freq_grid = np.meshgrid(times, frequencies)
            spectrogram_tensor = plot_to_tensor(time_grid, freq_grid, power, times, frequencies)
            spectrogram_tensors.append(spectrogram_tensor.unsqueeze(0))
        combined_tensor = torch.cat(spectrogram_tensors, dim=0)
        

        label = file['labels'][event_index]
        
        save_data = {
            'spectrogram': combined_tensor,
            'label': label
        }

        torch.save(save_data, output_dir / f'event_{event_index}.pt')
        print(f'Saved event {event_index} with shape {combined_tensor.shape} and label {label}')