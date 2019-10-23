#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 11:29:04 2019

@author: malrawi
"""

import torchvision 
import PIL.ImageChops as ImgChops
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms


# from torch.autograd import Variable

''' torchvision.datasets.Cityscapes(root, split='train', mode='fine', 
target_type='instance', transform=None, target_transform=None, transforms=None) '''

# valid_classes not needed
valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 
                 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]

all_classes = valid_classes + void_classes
   

class CityScapes(Dataset):
    def __init__(self, folder_name='./data', split ='train', transform_cty = None, 
                 transform_data=None):        
                
        self.data = torchvision.datasets.cityscapes.Cityscapes('./data/',
                                               split=split,
                                               target_type=['semantic'],
                                               transform = transform_cty,
                                               target_transform = transform_cty
                                               )        
        self.transform = transform_data
        self.valid_classes = valid_classes
        self.split = split

    def __getitem__(self, idx):   
        item_A, seg = self.data[idx]                        
        available_classes = list( set(seg.getdata()) ) # returns the pixel values, each representing a class        
        # print(available_classes) # diagnostic
        if self.split !='train': # as the documentation sayd, for void classes ignor_in_evl             
            available_classes = [entry for entry in available_classes if entry not in void_classes]                            
            classes_id = [valid_classes.index(n) for n in available_classes ] 
        else:
            classes_id = [all_classes.index(n) for n in available_classes ] 
            
        
        no_classes = len(available_classes)               
        
        item_B = torch.zeros((no_classes, 3, seg.size[1], seg.size[0] ), dtype = torch.float32)
        
        for ii, class_id in enumerate(available_classes):            
            mask = seg.point(lambda p: p == class_id and 255)     
            rr, gg, bb = item_A.split()    
            cr = ImgChops.multiply(rr, mask)
            cg = ImgChops.multiply(gg, mask)
            cb = ImgChops.multiply(bb, mask)
            dd = Image.merge("RGB",(cr,cg,cb))                        
            if self.transform:
                dd = self.transform(dd)  
            item_B[ii,:,:,:] = dd
                    
        if self.transform:
                item_A = self.transform(item_A)  
        
        return {"A": item_A, "B": item_B, "classes_id": classes_id, "available_classes": available_classes}  
        

    def __len__(self):
        return len(self.data)


transform_data = transforms.Compose([    
   transforms.Resize(( 256, 256), Image.NEAREST),    
#    transforms.Resize(int(256 * 1.12), Image.NEAREST),
#    transforms.RandomCrop((256, 256)),
#    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
)

transform_cty = transforms.Compose([    
   transforms.Resize(( 256, 256), Image.NEAREST)    
]
)


# x = CityScapes(split='val', transform_cty = transform_cty, transform_data = transform_data)
               


#print(x[1]['classes_id'])
#print(x[1]['available_classes'])

#dataloader = DataLoader(
#    x,  
#    batch_size=1,
#    shuffle=True,
#    num_workers = 4,
#)

#cuda=True
#Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
#
#for i, batch  in enumerate(dataloader):      
#    print(i)
#    # Set model input
#    real_A = Variable(batch["A"].type(Tensor)) # this should be one image                                      
#    real_B = Variable(batch["B"].type(Tensor)) # this should be a total of no_classes  = 'no_models' images
#    
    
    