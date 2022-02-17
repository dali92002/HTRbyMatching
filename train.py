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
from load_data import load_data
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
get_error_rate = htr_utils.get_error_rate
BATCH_SIZE = options.batch_size

SHOTS = options.shots
TRAIN_TYPE = options.train_type
root =  options.data_path
root_txt = options.data_path+'annotation/train.txt'


shots_path = cipher+'_symbs'

val_lines_path = 'validation_data/'



 

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

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)




def get_gt():
    gt = []
    for x in list_lines[:]:
        line_gt = []
        f = open('eval_'+cipher+'/gt/'+x.split('.jpg')[0]+'.txt', "r")
        line = (f.read())
        f.close()

        gt.append(txt_to_int(line))
    return gt

def txt_to_int(text):
    res = []
    
    
    
    alpha_f = os.listdir(alphabet_path+'/'+cipher)
    # alpha_f.append(' ')   ################# borg ###########""
    # alpha_f = os.listdir('few5/old_arab_symbs5')
    text= text.split('\n')[0]
    # text = text.split(' ') ########################### copiale
    
    
    
    
    
    # print('text2 = ',text)
    for c in text:
        if c in ['?','[','b',']','a','k','O','I','A','\n','3','T','R','X']: #### borg
        # if c in ['\n',' ']:  #### runic
        # if c  in ['?','[','b',']','a','k','O','I','A','\n','3','T','R','X']:# ['\n','x','q','y','V','P','L',"#",'R','H','Z','N','B','K','(',')','F','M','G','T','C','D']: ### copiale
            continue
            
        else:
            if c==':':
                c='cl'
            if c=='.':
                c='dt'
            if c==',':
                c='cm'
            if c == ' ':
                res.append(-1)
            else:
                res.append(alpha_f.index(c))
    return (res)





if TRAIN_TYPE == 'fine_tune':
    model.load_state_dict(torch.load('models/omniglot.pth'))

    print("model loaded")

best_cer = 1


dataset,data_loader = load_data(BATCH_SIZE,SHOTS,root, root_txt)


print_fr = int(len(dataset)/BATCH_SIZE/4)
for epoch in range(0, 100):

    if epoch >-1:
        
        list_lines = os.listdir(val_lines_path+cipher)[:10]

        results = draw_and_read(model,list_lines,val_lines_path,cipher,SHOTS)
        gt = get_gt()

        predictions  = zid_read(results)[0]

        cer = get_error_rate(gt,predictions)[0]

        print('\n CER: ',cer)

        if cer<best_cer:
            best_cer = cer
            torch.save(model.state_dict(), 'models/best_model_'+cipher+'_.pth')

        print('\n best CER:', best_cer,'\n')

    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=print_fr)


a=414