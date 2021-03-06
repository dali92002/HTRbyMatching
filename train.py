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
val_data_path = options.val_data_path
root_txt = options.data_path+'annotation/train.txt'


shots_path = cipher+'_symbs'

val_lines_path = val_data_path+'/lines/'
val_text_path  = val_data_path+'/gt/'


 

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



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)




def get_gt():
    gt = []
    for x in list_lines[:]:
        f = open(val_text_path+cipher+'/'+x.split('.jpg')[0]+'.txt', "r")
        line = (f.read())
        f.close()

        gt.append(txt_to_int(line))
    return gt


def txt_to_int(text):
    res = []
    alpha_f = os.listdir(alphabet_path+'/'+cipher)
    text= text.split('\n')[0]
    text = text.split(' ')
    for c in text:
        if c not in alpha_f: #### borg
            res.append(-3)   #### if you want to ignore out of vocab symbols make it continue
        else:
            # a=41
            # if c==':':
            #     c='cl'
            # if c=='.':
            #     c='dt'
            # if c==',':
            #     c='cm'
            # # if c == ' ':   
            # #     res.append(-1)
            # else:
            res.append(alpha_f.index(c))
    return (res)





if TRAIN_TYPE == 'fine_tune':
    model.load_state_dict(torch.load('weights/omniglot.pth'))

    print("model loaded")

best_cer = 1
dataset,data_loader = load_data(BATCH_SIZE,SHOTS,root, root_txt)


print_fr = int(len(dataset)/BATCH_SIZE/4)

# training here
for epoch in range(0, 100):

    if epoch >-1:
        
        list_lines = os.listdir(val_lines_path+cipher)[:30]

        results = draw_and_read(model,list_lines,val_lines_path,cipher,SHOTS)
        gt = get_gt()

        predictions  = zid_read(results)[0]

        cer = get_error_rate(gt,predictions)[0]

        print('Validation CER: ',cer)

        if cer<best_cer:
            best_cer = cer
            if not os.path.exists('weights'):
                os.makedirs('weights')
            torch.save(model.state_dict(), 'weights/best_model_'+cipher+'_.pth')

        print('best Validation CER:', best_cer,'\n')

    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=print_fr)


a=414