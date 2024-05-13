import numpy as np
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
import tqdm
import torchvision
import torchvision.transforms as transforms
from vit import ViT
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch_loader import *
from bhutan_loader import *

def prepare_dataloaders(batch_size, dataset = remaining_dataset):
    # #IF SLEEP-EDF
    # batch_size = 128
    # train_size = int(0.8 * len(remaining_dataset))
    # test_size = len(remaining_dataset) - train_size

    # train_dataset, test_dataset = torch.utils.data.random_split(remaining_dataset,
    #                                                             [train_size, test_size])
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)#, num_workers=4)
    
    #IF BHUTAN
    batch_size = 25
    train_valid_size = int(0.85 * len(Bhutan_dataset))
    test_size = len(Bhutan_dataset) - train_valid_size
    generator1 = torch.Generator().manual_seed(0)
    train_valid_dataset, test_dataset = torch.utils.data.random_split(
    Bhutan_dataset, 
    [train_valid_size, test_size], 
    generator=generator1
    )

    train_size = int(0.70 * len(Bhutan_dataset))
    valid_size = len(train_valid_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(
    train_valid_dataset,
    [train_size, valid_size],
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False, num_workers=3)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=3)
    
    return train_loader, test_loader, valid_loader, train_dataset, test_dataset, valid_dataset

#IF BHUTAN, 19 channels and batch 25 with 'cls' and 'fixed', if SLEEP-EDF 3 channels and 128 batch with 'cls' and 'learnable'.
def main(image_size=(224,224), patch_size=(56,56), channels=19, 
         embed_dim=224, num_heads=4, num_layers=4, num_classes=5,
         pos_enc='fixed', pool='cls', dropout=0.3, fc_dim=None, 
         num_epochs=50, batch_size=25, lr=1e-3, warmup_steps=625,
         weight_decay=0.1, gradient_clipping=1
    ):

    out_dict = {'train_acc': [],
              'test_acc': [],
              'train_loss': [],
              'test_loss': [],
              'y_pred': [],
              'y_true': []}

    loss_function = nn.CrossEntropyLoss()

    train_iter, test_iter, valid_iter, train_set, test_set, valid_set = prepare_dataloaders(batch_size=batch_size, dataset = Bhutan_dataset)

    model = ViT(image_size=image_size, patch_size=patch_size, channels=channels, 
                embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
                pos_enc=pos_enc, pool=pool, dropout=dropout, fc_dim=fc_dim, 
                num_classes=num_classes
    )

    if torch.cuda.is_available():
        model = model.to('cuda')

    opt = torch.optim.AdamW(lr=lr, params=model.parameters(), weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / warmup_steps, 1.0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # training loop
    best_val_loss = 1e10
    for e in range(num_epochs):
        print(f'\n epoch {e}')
        model.train()
        train_loss = 0
        for image, label in tqdm.tqdm(train_iter):
            if torch.cuda.is_available():
                image, label = image.to('cuda'), label.to('cuda')
            opt.zero_grad()
            out, atten_matix_list = model(image)
            loss = loss_function(out, label)
            loss.backward()
            train_loss += loss.item()
            # if the total gradient vector has a length > 1, we clip it back down to 1.
            if gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            opt.step()
            sch.step()

        train_loss/=len(train_iter)

        val_loss = 0
        with torch.no_grad():
            model.eval()
            tot, cor= 0.0, 0.0
            y_true = []
            y_pred = []
            for image, label in test_iter:
                if torch.cuda.is_available():
                    image, label = image.to('cuda'), label.to('cuda')
                out, atten_matrix_list = model(image)
                loss = loss_function(out, label)
                val_loss += loss.item()
                out = out.argmax(dim=1)
                y_pred.extend(out.cpu().numpy())
                y_true.extend(label.cpu().numpy())
                tot += float(image.size(0))
                cor += float((label == out).sum().item())
            acc = cor / tot
            val_loss /= len(test_iter)
            std = 1 - accuracy_score(y_true, y_pred)
            sem = std / np.sqrt(len(y_true))
            sem_p = (sem / 1)
            print(f'-- train loss {train_loss:.3f} -- test accuracy {acc:.3f} -- test loss: {val_loss:.3f} -- SEM {sem_p * 100:.3f}')
            if val_loss <= best_val_loss:
                torch.save(model.state_dict(), 'model2_ViT_Bhutan.pth')
                best_val_loss = val_loss
            
        out_dict['y_pred'].append(y_pred)
        out_dict['y_true'].append(y_true)

        # save model
        torch.save(model, 'model2_ViT_Bhutan.pth')
    return out_dict

if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"]= str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    # set_seed(seed=1)  #set udelukkende seed hvis jeg ønsker at træne fra samme udgangspunkt (vægte starter fra samme sted).
    main()