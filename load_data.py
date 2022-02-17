from PIL import Image
import os
import torch
import sys
import cv2 
import transforms as T 
import utils
from configs import getOptions
import random

options = getOptions().parse()
cipher = 'borg'#options.cipher




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
            
            # if 'line' in filename:
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


# labels_path = 'posandtrue'
# neg_path = 'neg'

# except_neg_path = 'neg'
# #############################################################################################
# def readposneg():
#     L= {}
#     for key0 in os.listdir('few5/annotation/progressive/'+labels_path):
#         k=key0.split('.txt')[0]

#         L[k]={}

        
#         filename = "few5/"+cipher+"_lines/"+k.split('class')[0]
#         img = cv2.imread(filename)
#         (rows,cols) = img.shape[:2]


#         L[k]['filepath'] = filename
#         L[k]['class'] = k.split('class')[1]            
#         L[k]['width'] = cols
#         L[k]['height'] = rows
#         L[k]['bboxes'] = []
        
#         with open('few5/annotation/progressive/'+labels_path+'/'+key0) as f:
#             lines = f.readlines()
#         for line in lines:
            
#             x1 = line.split(',')[1]
#             y1 = line.split(',')[2]
#             x2 = line.split(',')[3]
#             y2 = line.split(',')[4]
#             pseudo = 'yes' in line.split(',')[7]

#             L[k]['bboxes'].append({'class': 1,'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2), 'score': 1,'pseudo':pseudo})
        
        
#         try:
#             with open('few5/annotation/progressive/'+neg_path+'/'+key0) as f:
#                 lines = f.readlines()
#         except:
#             with open('few5/annotation/progressive/'+except_neg_path+'/'+key0) as f:
#                 lines = f.readlines()
#         for line in lines:
            
#             x1 = line.split(',')[1]
#             y1 = line.split(',')[2]
#             x2 = line.split(',')[3]
#             y2 = line.split(',')[4]

#             L[k]['bboxes'].append({'class': 0,'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2), 'score': 1,'pseudo':pseudo})

#     return L

# #############################################################################################

    









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
        
#         self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
        
    def __getitem__(self, idx):
        # load images ad masks
#         img_path = os.path.join(self.root, "train", self.imgs[idx])

        img_path = self.imgs[idx][0]
        
        
        images_choices = os.listdir(self.root+cipher+'_symbs'+str(self.augment)+'/'+self.Xdata[img_path]['class']+'/') ###########


        try:
            img1 = Image.open(self.Xdata[img_path]['filepath']).convert("RGB")
        except:
            img1 = Image.open(self.Xdata[img_path]['filepath'].split('.png')[0]+'.jpg').convert("RGB")
        try:
            img2 = Image.open(self.root+cipher+'_symbs/'+self.Xdata[img_path]['class']+'/'+random.choice(images_choices).split('.png')[0]+'.jpg').convert("RGB")
        
        except:
            img2 = Image.open(self.root+cipher+'_symbs/'+self.Xdata[img_path]['class']+'/'+random.choice(images_choices).split('.jpg')[0]+'.jpg').convert("RGB")
        
        
        the_class = (self.Xdata[img_path]['class'])
        
        num_objs = len(self.Xdata[img_path]['bboxes'])
        boxes = []
        labels = []
        for j in range(num_objs):
            # if (self.Xdata[img_path]['bboxes'][j]['class'])==1:
            
            xmin = self.Xdata[img_path]['bboxes'][j]['x1']
            xmax = self.Xdata[img_path]['bboxes'][j]['x2']
            ymin = self.Xdata[img_path]['bboxes'][j]['y1']
            ymax = self.Xdata[img_path]['bboxes'][j]['y2']
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.Xdata[img_path]['bboxes'][j]['class'])#int(X[img_path]['bboxes'][j]['class']))
        # convert everything into a torch.Tensor
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
        
        
        
        bbox2 = []
        bbox2.append([0,0,224,224])
        bbox2 = torch.as_tensor(bbox2, dtype=torch.float32)
        area2 = (bbox2[:, 3] - bbox2[:, 1]) * (bbox2[:, 2] - bbox2[:, 0])
        # suppose all instances are not crowd
        iscrowd2 = torch.zeros((1,), dtype=torch.int64)
        
        
        target2 = {}
        target2["boxes"] = bbox2
        target2["labels"] = torch.as_tensor([1], dtype=torch.int64)
        target2["image_id"] = image_id
        target2["area"] = area2
        target2["iscrowd"] = iscrowd

        if self.transforms is not None:
            img1,target = self.transforms(img1,target)
            img2 = self.transforms(img2,target2)#Fsupp.to_tensor(img2)#

        
        return img1, img2[0], target 

    def __len__(self):
        return len(self.imgs)


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def load_data(batch_s,shots_number,root,txtfile):
    L = get_data2(txtfile,False)

    dataset_lab = readQuerySupport(root,L, shots_number,get_transform(train=False))
    data_loader = torch.utils.data.DataLoader(dataset_lab, batch_size=batch_s, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)
    
    return dataset_lab,data_loader


