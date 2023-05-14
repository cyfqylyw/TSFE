import argparse
from itertools import chain
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import vit_pytorch

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class WedIqaDataset(Dataset):
    def __init__(self, file_list, sim_label, transform=None):
        self.file_list = file_list
        self.transform = transform
        self.sim_label = sim_label

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = self.sim_label[idx]

        return img_transformed, label
    

def parse_args():
    parser = argparse.ArgumentParser(description="TSFE")

    parser.add_argument("--model_path", type=str,
                        default="./results/models", help="folder to save trained models")  # path to store model
    parser.add_argument("--loss_path", type=str,
                        default="./results/loss", help="folder to save trained loss")  # path to store loss
    parser.add_argument("--device", type=str,
                        default="cuda:0", help="device of GPU or CPU")

    args = parser.parse_args()
    return args


def main(args):
    # Training settings
    batch_size = 64
    epochs = 200
    lr = 1e-3
    gamma = 0.7
    seed = 42

    seed_everything(seed)
    device = args.device  # mps, cpu, cuda:6

    train_list = []
    path_pre = './exploration_database_and_code/distort_images/'

    for i in range(1, 4745):
        img_id = ('00000' + str(i))[-5:]
        for dist in ['gblur', 'gnoise', 'jpg', 'jp2k']:
            for level in range(1,6):
                path = path_pre + dist + '/' + dist + '_' + img_id + '_' + str(level)
                path += '.bmp' if dist in ['gblur', 'gnoise'] else '.jpg'
                train_list.append(path)

    labels_ssim = list(np.load('./results/wed_ssim.npy').reshape(-1,))  # ssim
    labels_fsim = list(np.load('./results/wed_fsim.npy').reshape(-1,))  # fsim
    labels_gmsd = list(np.load('./results/wed_gmsd_neg.npy').reshape(-1,))  # gmsd
    labels = [(x*y*(z**2))**(1/4) for x,y,z in zip(labels_ssim, labels_fsim, labels_gmsd)]

    # labels = list((np.load('./results/wed_ssim.npy').reshape(-1,)+np.load('./results/wed_fsim.npy').reshape(-1,))/2)  # (ssim + fsim) / 2


    for i in range(1, 4745):
        img_id = ('00000' + str(i))[-5:]
        path = './exploration_database_and_code/pristine_images/' + img_id + '.bmp'
        train_list.append(path)

    max_label = max(labels)
    new_label = [max_label+random.randint(1,1000)/1000*(1-max_label) for _ in range(4744)]
    labels = labels + new_label

    print(f"Train list: {len(train_list)}")
    print(f"Label list: {len(labels)}")


    train_list, test_list, train_label, test_label = train_test_split(train_list, labels, test_size=0.1, random_state=seed)

    print(f"Train Data: {len(train_list), len(train_label)}")
    print(f"Test Data: {len(test_list), len(test_label)}")


    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )    

    train_data = WedIqaDataset(train_list, train_label, transform=train_transforms)
    test_data = WedIqaDataset(test_list, test_label, transform=test_transforms)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    print(f"Train Data: {len(train_data), len(train_loader)}")
    print(f"Test Data: {len(test_data), len(test_loader)}")


    num_features = 256

    model_vit = vit_pytorch.SimpleViT(
        image_size = 256,
        patch_size = 32,
        num_classes = num_features,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048
    ).to(device)

    model_mlp = torch.nn.Sequential(
        torch.nn.Linear(num_features, 1),
        torch.nn.Flatten(0, 1)
    ).to(device)

    model_vit.load_state_dict(torch.load('./results/vit_models/sfg_vit_40_epoch.pkl'))
    model_mlp.load_state_dict(torch.load('./results/vit_models/sfg_mlp_40_epoch.pkl'))

    # loss function
    criterion = nn.MSELoss()
    # optimizer
    optimizer_vit = optim.SGD(model_vit.parameters(), lr=lr)
    optimizer_mlp = optim.SGD(model_mlp.parameters(), lr=lr)


    start = 40
    total_loss = []
    for epoch in range(epochs):
        epoch_loss = []

        for data, label in tqdm(train_loader):
            data = data.to(torch.float32).to(device)
            label = label.to(torch.float32).to(device)

            output = model_mlp(model_vit(data))
            loss = criterion(output, label)
            epoch_loss.append(loss)

            optimizer_vit.zero_grad()
            optimizer_mlp.zero_grad()

            loss.backward()

            optimizer_vit.step()
            optimizer_mlp.step()

        print(f"Epoch : {start+epoch+1} - loss : {sum(epoch_loss)/len(epoch_loss):.4f} \n")
        # print(epoch_loss)
        total_loss.append(epoch_loss)
        # show(total_loss)
        if epoch % 20 == 19:
            np.save(args.loss_path + '/sfg2_loss_' + str(start+epoch+1) + '_epoch.npy', torch.tensor(total_loss).cpu().numpy())
            torch.save(model_vit.state_dict(), args.model_path + '/sfg2_vit_'+str(start+epoch+1)+'_epoch.pkl')
            torch.save(model_mlp.state_dict(), args.model_path + '/sfg2_mlp_'+str(start+epoch+1)+'_epoch.pkl')


if __name__ == "__main__":
    args = parse_args()
    main(args)
