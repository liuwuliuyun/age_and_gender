import torch
from torchvision import models

model_ft = models.resnet18(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 20)

state_dict = torch.load('./weights/train_07_04/resnet18_Final.pth')

model_ft.load_state_dict(state_dict)

