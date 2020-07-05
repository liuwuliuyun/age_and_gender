import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class UTKDataset(Dataset):
    def __init__(self, dataset_dir, transform = None):
        self.images = []
        self.image_names = os.listdir(dataset_dir)
        self.age_list = []
        self.gender_list = []
        self.race_list = []

        for label in self.image_names:
            age, gender, race = list(map(int, label.split('.')[0].split('_')[:3]))
            age = torch.tensor([age])
            gender = torch.tensor([gender])
            race = torch.tensor([race])
            self.age_list.append(age)
            self.gender_list.append(gender)
            self.race_list.append(race)

        self.transform = transform
        for i in self.image_names:
            path = os.path.join(dataset_dir, i)
            img = Image.open(path)
            self.images.append(img.copy())
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        if self.transform:
            image = self.transform(image)
        '''
        [age] is an integer from 0 to 116, indicating the age
        [gender] is either 0 (male) or 1 (female)
        [race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
        '''
        return self.age_list[index], self.gender_list[index], self.race_list[index], image