# Bachelor Thesis

## To Access and Run sleep EDF
- Open EDF_Dataloader.ipynb
- Remember to select the correct path PSG and hypnogram file
- The spectrograms will be loaded as a .png file along with its label
- To run the models, please make sure the path for torch_loader fits your path

## To Access and Run Bhutan
- Open load_bhutan and select the path to the directory of NO_ICA Bhutan preprocessed data
- Run the code which generates stacked tensors for all channels along with the label in .pt file for each spectrogram.
- The Dataloader for the ViT can be found under loading_bhutan.py
