from PIL import Image
import os
import torch
import sys
import cv2 
import transforms as T 
import torchvision.transforms as torchT
import torchvision.transforms.functional as TF

import utils
from configs import getOptions
import random

options = getOptions().parse()
cipher = options.cipher #"synthetic"#
alphabet = options.alphabet
resizing =  options.resize


def get_data2(input_path,labeling=False):
    all_imgs = {}

    i = 1
    with open(input_path,'r') as f:
        print('Parsing annotation files')
        for line in f:
            sys.stdout.write('\r'+'idx=' + str(i))
            i += 1

            line_split = line.strip().split(',')
            if not labeling:
                (filename,x1,y1,x2,y2,class_name) = line_split
                pseudo = False
            else:
                (filename,x1,y1,x2,y2,class_name,pseudo_v) = line_split
                if pseudo_v=='no':
                    pseudo=False
                else:
                    pseudo=True
            
            # if i == 100:
            #     return all_imgs
            
            if filename+"class"+class_name not in all_imgs:
                all_imgs[filename+"class"+class_name] = {}
                try:
                    img = cv2.imread(filename.split('.png')[0]+'.jpg')
                    (rows,cols) = img.shape[:2]
                except:
                    img = cv2.imread(filename)#.split('.png')[0]+'.jpg')
                    (rows,cols) = img.shape[:2]
                all_imgs[filename+"class"+class_name]['filepath'] = filename
                all_imgs[filename+"class"+class_name]['class'] = class_name            
                all_imgs[filename+"class"+class_name]['width'] = cols
                all_imgs[filename+"class"+class_name]['height'] = rows
                all_imgs[filename+"class"+class_name]['bboxes'] = []
            
            all_imgs[filename+"class"+class_name]['bboxes'].append({'class': 1,'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2), 'score': 1,'pseudo':pseudo})
            for key in (all_imgs):
                if (key !=filename+"class"+class_name):
                    if(all_imgs[key]['filepath']==filename):
                        all_imgs[key]['bboxes'].append({'class': 0,'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})
                        for box in all_imgs[key]['bboxes']:
                            if(box['class']==1):
                                box2 = {'class': 0,'x1': box['x1'], 'x2': box['x2'], 'y1': box['y1'], 'y2': box['y2']}
                                if (box2 not in all_imgs[filename+"class"+class_name]['bboxes']):
                                    all_imgs[filename+"class"+class_name]['bboxes'].append(box2)
        
        return all_imgs



class readQuerySupport(object):
    def __init__(self, root,Xdata, augment, transforms):
        self.Xdata = Xdata
        self.root = root
        self.transforms = transforms
        self.augment = augment
        
        self.listimg = list(sorted(self.Xdata.keys()))
        self.imgs = []
        for i in range (len(self.listimg)):
            for c in range (augment):
                self.imgs.append([self.listimg[i],c])
    def __getitem__(self, idx):

        img_path = self.imgs[idx][0]
        
        
        images_choices = os.listdir(alphabet+'/'+cipher+'/'+self.Xdata[img_path]['class']+'/') ###########
        random.shuffle(images_choices)

        try:
            img1 = Image.open(self.Xdata[img_path]['filepath']).convert("RGB")
        except:
            img1 = Image.open(self.Xdata[img_path]['filepath'].split('.png')[0]+'.jpg').convert("RGB")
        try:
            img2 = Image.open(alphabet+'/'+cipher+'/'+self.Xdata[img_path]['class']+'/'+random.choice(images_choices).split('.png')[0]+'.jpg').convert("RGB")
        
        except:
            img2 = Image.open(alphabet+'/'+cipher+'/'+self.Xdata[img_path]['class']+'/'+random.choice(images_choices).split('.jpg')[0]+'.jpg').convert("RGB")
        
        image_size = img1.size
        
        if resizing:
            resize_factors = [2048/image_size[0], 128/image_size[1]]
            image_size = [2048,128]
        else:
            resize_factors = [1,1]
        num_objs = len(self.Xdata[img_path]['bboxes'])
        boxes = []
        labels = []
        for j in range(num_objs):
            xmin = int(self.Xdata[img_path]['bboxes'][j]['x1'] * resize_factors[0])
            xmax = int(self.Xdata[img_path]['bboxes'][j]['x2'] * resize_factors[0])
            ymin = int(self.Xdata[img_path]['bboxes'][j]['y1'] * resize_factors[1])
            ymax = int(self.Xdata[img_path]['bboxes'][j]['y2'] * resize_factors[1])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.Xdata[img_path]['bboxes'][j]['class'])

        if resizing:
            img1 = img1.resize((2048,128))
            img2 = img2.resize((128,128))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        
        

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        if self.transforms is not None:
            # transform image 1
            img1,target = self.transforms(img1,target)
            i, j, h, w  = torchT.RandomCrop.get_params(img1, output_size=(image_size[1]-8,image_size[0]))
            img1 = TF.crop(img1, i, j, h, w)
            img1 = torchT.Resize((image_size[1],image_size[0]))(img1)

            # transform image 2
            img2,_ = self.transforms(img2,None)#Fsupp.to_tensor(img2)#
            i, j, h, w = torchT.RandomCrop.get_params(img2, output_size=(image_size[1]-8,image_size[1]-8))
            img2 = TF.crop(img2, i, j, h, w)
            img2 = torchT.Resize((image_size[1],image_size[1]))(img2)
            img2 = torchT.RandomRotation(degrees=(-10, 10))(1-img2)
            img2 = 1-img2      

        return img1, img2, target 

    def __len__(self):
        return len(self.imgs)

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    
    return T.Compose(transforms)

def load_data(batch_s,shots_number,root,txtfile=None, L=None):
    if txtfile:
        L = get_data2(txtfile,False)

    dataset_lab = readQuerySupport(root,L, shots_number,get_transform(train=False))
    data_loader = torch.utils.data.DataLoader(dataset_lab, batch_size=batch_s, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
    
    return dataset_lab,data_loader


