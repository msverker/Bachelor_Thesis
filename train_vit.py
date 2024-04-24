import numpy as np
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
import tqdm

import torch
import torchvision
import torchvision.transforms as transforms
from vit import ViT
from sklearn.model_selection import train_test_split
from torch_loader import Data

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def select_two_classes_from_sleep_edf(dataset, classes):
    idx = (np.array(dataset.targets) == classes[0]) | (np.array(dataset.targets) == classes[1])
    dataset.targets = np.array(dataset.targets)[idx]
    dataset.targets[dataset.targets==classes[0]] = 0
    dataset.targets[dataset.targets==classes[1]] = 1
    dataset.targets= dataset.targets.tolist()  
    dataset.data = dataset.data[idx]
    return dataset

def prepare_dataloaders(batch_size, classes=[3, 7], dataset = Data):
    # TASK: Experiment with data augmentation
    train_transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    test_transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_set, test_set = train_test_split(dataset, test_size = 0.2, shuffle = True)
    

    # select two classes 
    #train_set = select_two_classes_from_sleep_edf(train_set, classes=classes)
    #test_set = select_two_classes_from_sleep_edf(test_set, classes=classes)


    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=True
    )
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False
    )
    return trainloader, testloader, train_set, test_set


def main(image_size=(128,128), patch_size=(16,16), channels=3, 
         embed_dim=128, num_heads=4, num_layers=4, num_classes=5,
         pos_enc='learnable', pool='cls', dropout=0.3, fc_dim=None, 
         num_epochs=20, batch_size=16, lr=1e-4, warmup_steps=625,
         weight_decay=1e-3, gradient_clipping=1
         
    ):

    loss_function = nn.CrossEntropyLoss()

    train_iter, test_iter, _, _ = prepare_dataloaders(batch_size=batch_size, dataset = Data)

    model = ViT(image_size=image_size, patch_size=patch_size, channels=channels, 
                embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
                pos_enc=pos_enc, pool=pool, dropout=dropout, fc_dim=fc_dim, 
                num_classes=num_classes
    )

    if torch.cuda.is_available():
        model = model.to('cuda')

    opt = torch.optim.AdamW(lr=lr, params=model.parameters(), weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / warmup_steps, 1.0))

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
            for image, label in test_iter:
                if torch.cuda.is_available():
                    image, label = image.to('cuda'), label.to('cuda')
                out, atten_matrix_list = model(image)
                loss = loss_function(out, label)
                val_loss += loss.item()
                out = out.argmax(dim=1)
                tot += float(image.size(0))
                cor += float((label == out).sum().item())
            acc = cor / tot
            val_loss /= len(test_iter)
            print(f'-- train loss {train_loss:.3f} -- validation accuracy {acc:.3f} -- validation loss: {val_loss:.3f}')
            if val_loss <= best_val_loss:
                torch.save(model.state_dict(), 'model.pth')
                best_val_loss = val_loss

        # save model
        torch.save(model, 'model.pth')


if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"]= str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    # set_seed(seed=1)  #set udelukkende seed hvis jeg ønsker at træne fra samme udgangspunkt (vægte starter fra samme sted).
    main()