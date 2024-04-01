import numpy as np
import pandas as pd
import cv2
from PIL import Image
from torch.utils.data import Dataset


def preprocess_label(unprocessed_label):
    if unprocessed_label == 1:
        return 0
    elif unprocessed_label in [2,3,4]:
        return 1
    else:
        raise ValueError('Invalid Label')
    
class DikoDataset(Dataset):
    def __init__(self, csv_file, transforms1 = None, transforms2 = None):
        self.df = pd.read_csv(csv_file)
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        path_object = self.df.loc[index]['Path']
        image_path = './OAI_Xray' + path_object 
        image = cv2.imread(image_path)
        image1 = image2 = Image.fromarray(image)
        
        if self.transforms1:
            image1 = self.transforms1(image1)
        if self.transforms2:
            image2 = self.transforms2(image2)
        
        # Get label
        label = self.df.loc[index]['Label']
        label = preprocess_label(label)
        return image1, image2, label
