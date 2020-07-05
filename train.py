import torch
import torch.nn as nn
import numpy as np
import torchvision
import time
import os
import copy
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
from data.UTKDataset import UTKDataset
from module.UTKLoss import MultiLoss

config = {
    'dataset_path':'/home/liuyun/dataset/UTKFace',
    'lr': 1e-4,
    'batch_size': 8,
    'num_epochs': 200,
    'save_dir': './weights/train_07_04/',
    'use_tb': False
}

if not os.path.exists(config['save_dir']):
    os.mkdir(config['save_dir'])


utk_data_augment = transforms.Compose([
    transforms.ColorJitter(brightness=0.08, contrast=0.05, saturation=0.05, hue=0.05),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Resize(128),
    transforms.ToTensor()
])

dataset = UTKDataset(config['dataset_path'], transform=utk_data_augment)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

# out features [13, 2, 5]
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 20)

criterion = MultiLoss()
optimizer = torch.optim.Adam(model_ft.parameters(), lr = config['lr'])

def train_model(model, criterion, optimizer, num_epochs = config['num_epochs'], use_tb = config['use_tb']):
    model = model.cuda()
    iter_num = 0
    if use_tb: writer = SummaryWriter()
    for epoch in range(1, num_epochs + 1):
        for age, gender, race, image in dataloader:
            start = time.time()
            image = image.cuda()
            optimizer.zero_grad()
            output = model(image)
            age_loss, gender_loss, race_loss = criterion(output, age, gender, race)
            loss = age_loss + gender_loss + race_loss
            loss.backward()
            optimizer.step()
            if use_tb:
                writer.add_scalar('Age Loss', age_loss.item(), iter_num)
                writer.add_scalar('Gender Loss', gender_loss.item(), iter_num)
                writer.add_scalar('Race Loss', race_loss.item(), iter_num)
                writer.add_scalar('Total Loss', age_loss.item() + gender_loss.item() + race_loss.item(), iter_num)
            print('Epoch:{}/{} || Age: {:.4f} Gen: {:.4f} Race: {:.4f} || Batchtime: {:.4f} s'.format(
                epoch, num_epochs, age_loss.item(), gender_loss.item(), race_loss.item(), time.time() - start
            ), end='\r')
            iter_num += 1
        print('\n')
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(config['save_dir'], 'resnet18_epoch'+ str(epoch) + '.pth'))
    torch.save(model.state_dict(), os.path.join(config['save_dir'], 'resnet18_Final.pth'))
    if use_tb: writer.close()

train_model(model_ft, criterion, optimizer)
