#!/usr/bin/python3

import torch
import torchvision

import threading
import time
import torch.nn.functional as F
from torch.nn import Sequential

import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import PIL.Image
import numpy as np

from lane_dataset import cs_dataset 

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()

def preprocess(image):
    device = torch.device('cuda')
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

TRANSFORMS = transforms.Compose([
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.Resize((66, 200)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dir_path = '/home/nano/workspace/imgs'
#dir_path = 'imgs'

dataset = cs_dataset(dir_path,transform=TRANSFORMS, random_hflip=False)

device = torch.device('cuda')

class cs_nvidia_model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.pass_size = 100
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3,24,kernel_size=5, stride=2),
            torch.nn.ELU(),
            torch.nn.Conv2d(24,36,kernel_size=5, stride=2),
            torch.nn.ELU(),
            torch.nn.Conv2d(36,48,kernel_size=5, stride=2),
            torch.nn.ELU(), 
            torch.nn.Conv2d(48,64,kernel_size=3),
            torch.nn.Dropout(0.2),
            torch.nn.Conv2d(64,64,kernel_size=3),
            torch.nn.ELU(),
            torch.nn.Flatten()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1152,100), # ??
            torch.nn.ELU(),
            torch.nn.Linear(100,50),
            torch.nn.ELU(),
            torch.nn.Linear(50,10),
            torch.nn.ELU(),
            torch.nn.Linear(10,1)
        )
        
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out


model = cs_nvidia_model()
model = model.to(device)
print(model)

# img = np.zeros((66,200,3),dtype=np.uint8)
# img = np.random.rand(66,200,3) * 255
# img = preprocess(img)
# out = model(img)
# print(out.shape)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train_eval(dataset,epochs, batch_no, is_training=True):

    global model, optimizer 
        
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_no,
        shuffle=True
    )

    if is_training:
        model = model.train()
    else:
        model = model.eval()

    while epochs > 0:
        i = 0
        sum_loss = 0.0
        error_count = 0.0
        for img_name, images, angles in iter(train_loader):
            # send data to device
            images = images.to(device)
            angles = angles.to(device)

            if is_training:
                # zero gradients of parameters
                optimizer.zero_grad()

            # execute model to get outputs
            outputs = model(images)
            
            loss = F.mse_loss(outputs, angles)

            if is_training:
                # run backpropogation to accumulate gradients
                loss.backward()
                # step optimizer to adjust parameters
                optimizer.step()

            # increment progress
            count = len(outputs.flatten())
            i += count
            
            progress = i / len(dataset)
            print(f'epoch ={epochs}, progress : {progress:.4f}, {i:04d} / {len(dataset):04d} batch_loss = {loss:.4f}')
        
        if is_training:
            epochs = epochs - 1
        else:
            break

if __name__ == "__main__" :

    import sys

    dir = '/home/nano/workspace/models/lane-det-cnn/'
    file = 'lane_follow_cnn.pth'

    epochs = 150
    if not sys.argv[1] == None :
        epochs = int(sys.argv[1])
 
    print('start training ')
    train_eval( dataset, epochs, batch_no=20, is_training=True)
    torch.save(model.state_dict(), dir + file)
        

