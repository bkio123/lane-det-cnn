#!/usr/bin/python3
import cv2

import time
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import PIL.Image
import numpy as np
import torchvision
import time
import atexit

device = torch.device('cuda')
working_dir = '/home/nano/workspace/models/lane-det-cnn/'

class cs_nvidia_model(torch.nn.Module):
    def __init__(self):
        super().__init__()

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

class lane_det_cnn_torch():

    
    def __init__(self):
        self.model = cs_nvidia_model()

    def preprocess(self,image):

        mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
       
        #image size change
        image = cv2.resize(image,(200,66))

        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(device)
        image.sub_(mean[:, None, None]).div_(std[:, None, None])
        return image[None, ...]

    def load_model(self, file):
        
        self.model.load_state_dict(torch.load( working_dir + file))
        #self.model.load_state_dict(torch.load(file))

        self.model = self.model.to(device)
        self.model.eval()

    def img_to_angle(self,image):
        image = self.preprocess(image)
        output = self.model(image)
        x = float(output[0])
        angle = int(x * 900)
        return angle


if __name__ == '__main__' :
    
    img = cv2.imread('000.jpg')
    det = lane_det_cnn_torch()

    det.load_model('lane_follow_cnn.pth')
    print('load model')

    angle = det.img_to_angle(img)

    print(f' angle = {angle} ')

