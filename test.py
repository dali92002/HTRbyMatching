import torch
from torch import nn
import torchvision
from src.faster_rcnn import FastRCNNPredictor ,TwoMLPHead
import torchvision
from   src.faster_rcnn import FasterRCNN
from src.rpn import AnchorGenerator
import torchvision
import src.transforms as T 
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.engine import train_one_epoch, evaluate
import src.utils
import sys
import csv
import cv2 
import os
import random
import numpy as np
import PIL

from configs import getOptions
import htr_utils

options = getOptions().parse()

cipher = options.cipher

alphabet_path = options.alphabet

lines_path = options.lines

output_path = options.output

shots_number = options.shots
threshold = options.thresh

testing_model = options.testing_model


draw_and_read = htr_utils.draw_and_read
zid_read = htr_utils.zid_read
inttosymbs = htr_utils.inttosymbs


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2

backbone = torchvision.models.vgg16(pretrained=True).features
backbone.out_channels = 512 #128

anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

roi_ouput_size = 7
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                output_size=roi_ouput_size,
                                                sampling_ratio=2)
model = FasterRCNN(backbone,
                   num_classes=num_classes,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)

backbone_output_size = 512

in_channels = 512
in_channels2 = backbone_output_size*roi_ouput_size**2


model.roi_heads.box_predictor = FastRCNNPredictor(in_channels, num_classes)
model.roi_heads.box_head = TwoMLPHead(in_channels2, in_channels)


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)




# Load the model 
model.load_state_dict(torch.load(testing_model))


# alphabet_fo = alphabet_path


list_lines = os.listdir(lines_path+'/'+cipher)[:]

results = draw_and_read(model,list_lines,lines_path,cipher,shots_number)


#######################################""
predictions, pred_boxes  = zid_read(results)

pred_lines = inttosymbs(predictions,cipher)



### visialize-results
for ln,pl,bx in zip(list_lines,pred_lines,pred_boxes):

    if not os.path.exists(output_path+'/'+cipher+'/text'):
        os.makedirs(output_path+'/'+cipher+'/text')
    if not os.path.exists(output_path+'/'+cipher+'/boxes'):
        os.makedirs(output_path+'/'+cipher+'/boxes')
    if not os.path.exists(output_path+'/'+cipher+'/images'):
        os.makedirs(output_path+'/'+cipher+'/images')
    
    f=open(output_path +'/'+cipher+ '/text/'+ln+'.txt','w')
    f.write(pl)
    f.close()

    im = cv2.imread(lines_path+'/'+cipher+'/'+ln)
    masks = np.ones((im.shape[0],im.shape[1],3)) * 255
    masks  = masks.astype(np.uint8)

    text = np.ones((60,im.shape[1],3)) * 255
    text  = text.astype(np.uint8)
    
    
    f=open(output_path + '/'+cipher+'/boxes/'+ln+'.txt','w')
    
    pline = pl.split(' ')
    i = 0
    
    for b in range (0,len(bx),2):
        c1 = random.randint(0,255)
        c2 = random.randint(0,255)
        c3 = random.randint(0,255)

        f.write((str(bx[b])+','+str(bx[b+1])+'\n'))
        cv2.rectangle(masks, (bx[b],0), (bx[b+1], 105), (c1,c2,c3),-1)    
        clas = pline[i]
        i+=1
        cv2.putText(text,clas,  (bx[b]+ int((bx[b+1]-bx[b])/3),40),  cv2.FONT_HERSHEY_SIMPLEX, 0.7,(c1,c2,c3), 2)
   

    res = cv2.addWeighted(im,0.8,masks,0.2,0)


    vis_concatenate = np.concatenate((res, text), axis=0)

    cv2.imwrite(output_path + '/'+cipher+'/images/'+ln,vis_concatenate)
    