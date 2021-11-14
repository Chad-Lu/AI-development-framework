from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from skimage import io, transform
import os
import torch
from torchvision import transforms
import numpy as np
import torchvision.transforms as trns
from scipy.io import loadmat
from PIL import Image
import os
import pandas as pd



class Mango_data_df_test(Dataset):

    def __init__(self, root_dir, transform=None):


        #self.roofs_frame = pd.read_csv(root_dir,encoding="unicode_escape")
        self.root_dir = root_dir
        self.transform = transform


    def __len__(self):
        return len(self.roofs_frame)

    def __getitem__(self, idx):

        #img_name = self.root_dir + "sample_image//" + self.roofs_frame.iloc[idx, 0] #image_path
        #print('idx_num: ',idx)
        img_name = os.path.join(self.root_dir,self.roofs_frame.iloc[idx, 0])  #,'sample_image//'
        #print('path: ',img_name)
        image = Image.open(img_name)
        #image = io.imread(img_name)
        #image = Image.fromarray(image)
        #print(type(image))
        #landmarks = self.roofs_frame.iloc[idx, 1] #image_label
        #print(landmarks)
        '''
        if 'B' in landmarks:
            label = 1  # B級
        elif 'C' in landmarks:
            label = 2  # C級
        else:
            label = 0  # A級
        '''
        #print('label: ',label)

        #sample = {'image': image, 'label': label}
        #print(sample)

        if self.transform:
            out_image = self.transform(image)

        return out_image