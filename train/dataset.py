from torch.utils.data import Dataset
import pandas as pd
import torch
from PIL import Image
import clip
import os
import ast

class HardNegDataWithAffirmSet(Dataset):
    def __init__(self,transform=None):
        super().__init__()
        self.csv = pd.read_csv("/path/to/fine-tuning dataset")
        self.transform = transform

    def __getitem__(self, index):
        row_data = self.csv.iloc[index]
        image_path = row_data['image_path']
        original_caption = row_data['caption']

        negsub = row_data['negsub']
        negpred = ast.literal_eval(row_data['negpred'])
        negadv = ast.literal_eval(row_data['negadv'])
        
        negpred_caption = negpred['original']
        negpred_affirm = negpred['processed']
        negadv_caption = negadv['original']
        negadv_affirm = negadv['processed']
    
        if self.transform is not None:
            image = self.transform(Image.open(image_path))
    
        return image,original_caption,negsub,negpred_caption,negpred_affirm,negadv_caption,negadv_affirm

    def __len__(self):
        return len(self.csv)


class EvalDataSet(Dataset):
    def __init__(self,transform=None):
        super().__init__()
        self.csv = pd.read_csv("path/to/SAN-Bench matching csv")
        self.transform = transform


    def __getitem__(self, index):
        row_data = self.csv.iloc[index]
        image_path = row_data['image_path']
        image_path = os.path.join("/path/to/flickr30k_images",image_path)
        original_caption = row_data['caption']

        negsub = row_data['negsub']
        negpred = row_data['negpred']
        negadv = row_data['negadv']

        if self.transform is not None:
            image = self.transform(Image.open(image_path))
        else:
            image = Image.open(image_path)

        return image,original_caption ,negsub,negpred,negadv
    

    def __len__(self):
        return len(self.csv)
    



class RetrievalSet(Dataset):
    def __init__(self,transform=None):
        super().__init__()
        self.csv = pd.read_csv("/path/to/retrieval.csv")
        self.transform = transform
        self.image_path = "path/to/synthetic images"

    def __getitem__(self, index):
        row_data = self.csv.iloc[index]
        original_caption = row_data['Pos']
        images=  []
        negsub = row_data['Negsub']
        negpred = row_data['Negpred']
        negadv = row_data['Negadv']
        for i in range(4*index,4*index+4):
            image_path = os.path.join(self.image_path,f"{i}.png")
            if not os.path.exists(image_path):
                image_path = os.path.join(self.image_path,f"{i}.jpg")
            if self.transform is not None:
                image = self.transform(Image.open(image_path))
                images.append(image)

        return images,original_caption ,negsub,negpred,negadv
    

    def __len__(self):
        return len(self.csv)

