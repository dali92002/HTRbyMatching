import os
import random
import numpy as np
import PIL

import torch
from PIL import Image
import pickle
from torch import nn
import torchvision
from src.faster_rcnn import FastRCNNPredictor ,TwoMLPHead

import torchvision
from   src.faster_rcnn import FasterRCNN
from src.rpn import AnchorGenerator

from torchvision.transforms import functional as Fsupp

import torchvision

import transforms as T 
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

from src.engine import train_one_epoch, evaluate
import utils
from tqdm import tqdm
import sys
import csv
import cv2 
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import editdistance
from load_data import get_data2, load_data



if not os.path.exists('few5/annotation/progressive/positives/'):
    os.makedirs('few5/annotation/progressive/positives/')
if not os.path.exists('few5/annotation/progressive/negatives/'):
    os.makedirs('few5/annotation/progressive/negatives/')


positives = os.listdir('few5/annotation/progressive/positives/')
negatives = os.listdir('few5/annotation/progressive/negatives/')

print(len(positives)) 
for pos in positives:
    os.remove('few5/annotation/progressive/positives/'+pos)
for neg in negatives:
    os.remove('few5/annotation/progressive/negatives/'+neg)

print("Old removed")



def drawboxes(img_path,x1,x2,y1,y2,cls0,color):
    font = ImageFont.truetype("arial.ttf", 35)

    image = Image.open(img_path)# fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    
    draw = ImageDraw.Draw(image)
    
    draw.rectangle([(x1,y1), (x2,y2)], fill=None,outline='green')
    draw.text((x1,y1), cls0 ,fill=color,font=font)

    return image


def readposneg():
    L= {}
    for key0 in os.listdir('few5/annotation/progressive/positives'):
        k=key0.split('.txt')[0]

        L[k]={}

        try:
            filename = "few5/borg_lines/"+k.split('class')[0]
            img = cv2.imread(filename)
            (rows,cols) = img.shape[:2]
        except:
            filename = "few5/synthetic/"+k.split('class')[0]
            img = cv2.imread(filename)
            (rows,cols) = img.shape[:2]

        L[k]['filepath'] = filename
        L[k]['class'] = k.split('class')[1]            
        L[k]['width'] = cols
        L[k]['height'] = rows
        L[k]['bboxes'] = []
        
        with open('few5/annotation/progressive/positives/'+key0) as f:
            lines = f.readlines()
        for line in lines:
            
            x1 = line.split(',')[1]
            y1 = line.split(',')[2]
            x2 = line.split(',')[3]
            y2 = line.split(',')[4]
            pseudo = 'yes' in line.split(',')[7]

            L[k]['bboxes'].append({'class': 1,'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2), 'score': 1,'pseudo':pseudo})
        
        
        
        with open('few5/annotation/progressive/negatives/'+key0) as f:
            lines = f.readlines()
        for line in lines:
            
            x1 = line.split(',')[1]
            y1 = line.split(',')[2]
            x2 = line.split(',')[3]
            y2 = line.split(',')[4]

            L[k]['bboxes'].append({'class': 0,'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2), 'score': 1,'pseudo':pseudo})

    return L


cipher = 'borg'
X = get_data2('few5/annotation/synthetic.txt')
all_classes = os.listdir('alphabet/'+cipher)

lab = np.zeros(len(all_classes))+5

starting_lines=set()

X_lab={}
X_unlab={}
for x in X:
    if lab[all_classes.index(X[x]['class'])] >= 0 :
        X_lab.update({x:X[x]})
        
        line = X[x]['filepath'].split('few5/synthetic/')[1]
        starting_lines.add(line)
        
        lab[all_classes.index(X[x]['class'])]-=1

print('\nNumer of starting lines:', len(starting_lines))

f0=open('few5/annotation/starting_lines.txt',"w")

with open('few5/annotation/synthetic.txt') as f:
    ann_lines = f.readlines()
for ann_l in range (len(ann_lines)):
    ann_line = ann_lines[ann_l]
    if ann_line.split(',')[0].split('few5/synthetic/')[1] in starting_lines:
        f0.write(ann_line)
f0.close()


lines = os.listdir("few5/"+cipher+"_lines/")
alphabet = os.listdir('alphabet/'+cipher)




X_lab = get_data2('few5/annotation/starting_lines.txt')


def annotate(epoch):

    for key in X_lab.keys():
        try:
            f1= open("few5/annotation/progressive/positives/"+str(key.split("borg_lines/")[1])+".txt","w")
            f2 = open("few5/annotation/progressive/negatives/"+str(key.split("borg_lines/")[1])+".txt","w")
        except:
            f1= open("few5/annotation/progressive/positives/"+str(key.split("synthetic/")[1])+".txt","w")
            f2 = open("few5/annotation/progressive/negatives/"+str(key.split("synthetic/")[1])+".txt","w")
        for box in X_lab[key]['bboxes']:
            
            box_class = box['class']
            
            
            filepath  = X_lab[key]['filepath']
            
            x1 = int(box['x1'])
            x2 = int(box['x2'])
            y1 = int(box['y1'])
            y2 = int(box['y2'])

            className = X_lab[key]['class']
            pseudo='no'
            if box_class == 1: 
                f1.write(filepath + ',' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + str(className) +','+str(box_class)+','+ pseudo+'\n')
            else:
                f2.write(filepath + ',' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + str(className) +','+str(box_class)+','+ pseudo+'\n')
        f2.close()
        f1.close()

annotate(0)
L = readposneg()

def showlabels(key, X_labels,epo):
    # for key in X_labels.keys():
    img_path = X_labels[key]['filepath']
    saving = img_path.split('lines/')[1]+'_class_'+X_labels[key]['class']+'.jpg'

    for bbox in X_labels[key]['bboxes']:
        if bbox['class']==1 and bbox['pseudo']:
            color="blue"
        else:
            if bbox['class']==0:
                color = "red"
            # else:
            #     color="red" 
            
        x1 = bbox['x1']
        x2 = bbox['x2']
        y1 = bbox['y1']
        y2 = bbox['y2']
        
        

        cls0 = X_labels[key]['class']
        if not os.path.exists('labels/epoch_'+str(epo)+'/'):
            os.makedirs('labels/epoch_'+str(epo)+'/')
        drawboxes(img_path,x1,x2,y1,y2,cls0,color).save('labels/epoch_'+str(epo)+'/'+saving)
        img_path = 'labels/epoch_'+str(epo)+'/'+saving


def show_new(X_labels,epo):
    for key in X_labels.keys():
        
        for bbox in X_labels[key]['bboxes']:
            if bbox['class']==1 and bbox['pseudo']:
                showlabels(key,X_labels,epo)
                a=1212
    

curr_extend = 30
number_box=0



 

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

thresh = 0.4
def validat(model,test_data,epoch,same_data=False):
    torch.save(model.state_dict(), 'model/model_epoch_'+str(epoch)+'.pth')
    model.eval()

    for t in range (len(test_data)):
        img1, img2, _ = test_data[t]

        with torch.no_grad():
            prediction = model([img1.to(device)],[img2.to(device)])
        draw = drawboxes(img1,prediction[0],thresh=thresh)
        image2 = Image.fromarray(img2.mul(255).permute(1, 2, 0).byte().numpy())
        image_blanc = Image.new('RGB', (20, 105), (0, 0, 255))
        imgs_comb = np.hstack( (image2,image_blanc,draw) )

        imgs_comb = PIL.Image.fromarray( imgs_comb)
        if (same_data):
            if not os.path.exists('Results_same/epoch_'+str(epoch)):
                os.makedirs('Results_same/epoch_'+str(epoch))
            imgs_comb.save( 'Results_same/epoch_'+str(epoch)+'/image_'+str(t+1)+'.jpg' )  
        else: 
            if not os.path.exists('Results/epoch_'+str(epoch)):
                os.makedirs('Results/epoch_'+str(epoch))
            imgs_comb.save( 'Results/epoch_'+str(epoch)+'/image_'+str(t+1)+'.jpg' )  




def same_box(x1b1,x2b1,x1b2,x2b2):
    if x1b1<x1b2:
        aux = x1b1
        x1b1 = x1b2
        x1b2 = aux
        
        
        aux = x2b1
        x2b1 = x2b2
        x2b2=aux
    x1b1 = int(x1b1)
    x2b1 = int(x2b1)
    x1b2 = int(x1b2)
    x2b2 = int(x2b2)
    
    if x1b1 in range(x1b2,x2b2):
        if x2b2-x1b1 > 0.7 * (x2b1-x1b2):
            return True
        else:
            return False
    else:
        return False



def add_label(line_name,symb,box,score_box):
    global number_box
    to_add = True


    X_l1 = readposneg()

    to_check = [element for element in os.listdir("few5/annotation/progressive/positives/") if line_name in element]

    xr1 = int(box[0])
    x2r = int(box[1])

    for check_txt in to_check:

        with open("few5/annotation/progressive/positives/"+check_txt) as fps:
            ps_lines = fps.readlines()

        for psline in ps_lines:
            x1p = int(psline.split(',')[1])
            x2p = int(psline.split(',')[3])

            if (same_box(xr1,x2r,x1p,x2p)):
                to_add=False
 
    if to_add:
        filepath = 'few5/borg_lines/'+line_name
        x1 =  box[0]
        x2 =  box[1]
        y1 = 0
        y2 = 105

        if line_name+"class"+symb+'.txt' not in os.listdir('few5/annotation/progressive/positives/'):
            f= open("few5/annotation/progressive/positives/"+line_name+"class"+symb+'.txt',"w")
            f.write(filepath + ',' +  str(int(x1)) + ',' + str(y1) + ',' + str(int(x2)) + ',' + str(y2) + ',' + str(symb) +','+str(1)+','+ "yes"+'\n')
            f.close()
        else:
            f= open("few5/annotation/progressive/positives/"+line_name+"class"+symb+'.txt',"a")
            f.write(filepath + ',' +  str(int(x1)) + ',' + str(y1) + ',' + str(int(x2)) + ',' + str(y2) + ',' + str(symb) +','+str(1)+','+ "yes"+'\n')
            f.close()            

        

    
    return to_add

def add_negatives(label,negatives):
    line_name = label[0]
    symb = label[1]
    
    if line_name+"class"+symb+'.txt'  in os.listdir('few5/annotation/progressive/negatives/'):
        with open("few5/annotation/progressive/negatives/"+line_name+"class"+symb+'.txt') as fps:
            ps_lines = fps.readlines()
        
        f= open("few5/annotation/progressive/negatives/"+line_name+"class"+symb+'.txt',"w")
        
        for psline in ps_lines:
            x1p = int(psline.split(',')[1])
            x2p = int(psline.split(',')[3])

            if not(same_box(label[2][0],label[2][1],x1p,x2p)):
                f.write(psline)
        
        f.close()
    else:    
        # clean negatives
        negative_boxes = []
        f= open("few5/annotation/progressive/negatives/"+line_name+"class"+symb+'.txt',"w")
        
        positives_neg = []
        for lab in negatives:
            if lab[1]==label[1]:
                positives_neg.append([lab[2][0],lab[2][1]])

        for lab in negatives:
            
            
            adding = True
            for box in negative_boxes:
                if same_box(lab[2][0],lab[2][1],box[0],box[1]):
                    adding = False
            for box in positives_neg:
                if same_box(lab[2][0],lab[2][1],box[0],box[1]):
                    adding = False
            if adding:
                negative_boxes.append([lab[2][0],lab[2][1]])
                filepath = 'few5/borg_lines/'+line_name
                
                x1 =  lab[2][0]
                x2 =  lab[2][1]
                y1 = 0
                y2 = 105

                f.write(filepath + ',' + str(int(x1)) + ',' + str(y1) + ',' + str(int(x2)) + ',' + str(y2) + ',' + str(symb) +','+str(0)+','+ "yes"+'\n')
        f.close()



def select_new_labels(X_l,conf,detect_conf,starting = False, shots=10,the_epoch=0):
    new_lab_list=[]
    
    model.eval()
    i = 1
    for line_name in lines[:]:
        
        
        sys.stdout.write('\r'+'searching for labels=' + str(i))
        i += 1
        
        if True:
            line = Image.open('few5/borg_lines/'+line_name).convert("RGB")
            line = Fsupp.to_tensor(line)


            for symb in alphabet:

                for img2path in (os.listdir(shots_path+'/'+symb)):
                    founded_boxes_set = []
                    if '.png' in img2path:
                        img2path = img2path.split('.png')[0]+'.jpg'
                    img2 = Image.open(shots_path+'/'+symb+'/'+img2path).convert("RGB")
                    img2 = Fsupp.to_tensor(img2)
                    # choices.pop(0)
                    with torch.no_grad():
                        _ = model([line.to(device)],[img2.to(device)])
                    prediction = _[0]


                    for p in range (len(prediction['scores'])):
                        if prediction['scores'][p].item() >=detect_conf:

                            box = [prediction['boxes'][p][0].item(), prediction['boxes'][p][2].item(), prediction['boxes'][p][1].item(), prediction['boxes'][p][3].item()]#prediction['boxes'][p].item()
                            score_box = prediction['scores'][p].item()
                            
                            add_the_f_box = True

                            for fbox in founded_boxes_set:
                                if  same_box(fbox[0],fbox[1], box[0],box[1]):
                                    add_the_f_box = False
                                    founded_boxes_set[founded_boxes_set.index(fbox)][2] = max(score_box,fbox[2])
                            if add_the_f_box:
                                founded_boxes_set.append([box[0],box[1],score_box])
                for fbox in founded_boxes_set:
                    new_lab_list.append([line_name,symb,[fbox[0],fbox[1],0,105],fbox[2]])

    new_lab_list = sorted(new_lab_list, reverse = True, key=lambda x: x[3])
    
    c_l = 0
    c_added = 0
    
    curr_len = len_boxes(X_l)

    if starting:
        ext = 300
    else:
        ext = curr_len*0.2
    

    print('\ndetected boxes: ',len(new_lab_list))

    if True: #else
        alpha2 = alphabet_fo[:]
        alpha2 = list(np.repeat(alpha2, 2))
        i2=0
        while(c_l< len(new_lab_list) and new_lab_list[c_l][3]>conf and c_added<ext):#c_added<curr_len*ext and c_l< len(new_lab_list)):
            sys.stdout.write('\r'+'processing detected boxes:' + str(i2))
            i2 += 1
            

            if (add_label (new_lab_list[c_l][0],new_lab_list[c_l][1],new_lab_list[c_l][2],new_lab_list[c_l][3])):
                c_added = c_added+ 1
                
                the_label = new_lab_list[c_l]
                the_negatives= []

                for labn in new_lab_list:
                    if the_label[0] == labn[0]:
                        
                        the_negatives.append(labn)

                add_negatives(the_label,the_negatives)
                # if the_epoch <=50:
                #     alpha2.remove(new_lab_list[c_l][1])
            # else:
            #     true_extra.append([new_lab_list[c_l][0],new_lab_list[c_l][2]])

            c_l += 1

def verify(lab,true_labs):
    v = True
    for t_lab in true_labs:
        if t_lab[0]==lab[0] and same_box(t_lab[1][0],t_lab[1][1],lab[1][0],lab[1][1]):
            v=False
    return v

def len_boxes(X_lb):
    n_boxes = 0
    for keys_lb in X_lb.keys():
        for box in X_lb[keys_lb]['bboxes']:
            if (box['class']==1):
                n_boxes+=1
    return n_boxes

    
def drawprobs(img1,shots,st_ch,en_ch):
    mat_size  = 100
    
    image_hline = Image.new('RGB', (img1.size()[2]+5+mat_size, 5), (0, 0, 255))
    image1 = Image.fromarray(img1.mul(255).permute(1, 2, 0).byte().numpy())
    image_f = Image.new('RGB', (mat_size, 105), (255, 255, 255))
    image_vline = Image.new('RGB', (5, 105), (0, 0, 255))
    imgs_comb = np.hstack( (image_f,image_vline,image1) )

    thresh = 0.4
    
    font = ImageFont.truetype("arial.ttf", 25)
    
    Pro_matrix = np.zeros((en_ch-st_ch +1,img1.size()[2]))
    last_max = 0
    p_c = 0
    for i in range (len(alphabet_fo)):
        Matrix  = torch.zeros((3,mat_size,img1.size()[2]))
        choices  = list(range(1,11))
        for img2path in (os.listdir('eval_borg/'+shots_path+'/'+alphabet_fo[i])):
            
            img2 = Image.open(alphabet_fo+'/'+alphabet_fo[i]+'/'+img2path.split('.png')[0]+'.jpg').convert("RGB")
            img2 = Fsupp.to_tensor(img2)
            choices.pop(0)
            with torch.no_grad():
                _ = model([img1.to(device)],[img2.to(device)])
            _ = _[0]
            
            
        
            for  box,lab in zip (_['boxes'],range(_['scores'].size()[0])):
                if (_['scores'][lab].item()>thresh):
                    Mat = torch.zeros((3,mat_size,int(box[2].item())-int(box[0].item()))) + _['scores'][lab].item()
                    Matrix[:,:,int(box[0].item()):int(box[2].item()) ] = torch.max(Mat,Matrix[:,:,int(box[0].item()):int(box[2].item()) ]) 

                    Pmat = np.zeros((1,int(box[2].item())-int(box[0].item()))) + _['scores'][lab].item() 
                    Pro_matrix[p_c,int(box[0].item()):int(box[2].item())] = np.maximum(Pmat,Pro_matrix[p_c,int(box[0].item()):int(box[2].item())])
        p_c = p_c+1
        
        Matrix_draw = Image.fromarray(Matrix.mul(255/1).permute(1, 2, 0).byte().numpy())
        
        
        draw = ImageDraw.Draw(Matrix_draw,mode='RGB')
        
        
        pa = 30

        for d in range(0,img1.size()[2],pa):
            
            if (torch.max(Matrix[:,:,d:d+pa])!=last_max) and torch.max(Matrix[:,:,d:d+pa])>0:
                draw.text((d+20, 35), "%.2f" % (torch.max(Matrix[:,:,d:d+pa])) ,fill='red',font=font)
                last_max = torch.max(Matrix[:,:,d:d+pa])
            

            
        image2 = Image.fromarray(img2.mul(255).permute(1, 2, 0).byte().numpy())
        
        image_vline = Image.new('RGB', (5, mat_size), (0, 0, 255))
        
        img_comb = np.hstack( (image2.resize((mat_size,mat_size)),image_vline,Matrix_draw) )
        imgs_comb = np.vstack( (imgs_comb,image_hline,img_comb) )
        

    imgs_comb = PIL.Image.fromarray( imgs_comb)
    return imgs_comb, Pro_matrix

def read_sp_char(matrix,thr,conf = 0.3):
    maxs = matrix.max(axis=0)
    listchar = []
    occ = 0
    lastone=0
    last_max = 0
    
    sp_th=20
    
    for z in range (matrix.shape[1]):
        if z<matrix.shape[1]-sp_th-1:
            if (np.sum(matrix[:,z:z+sp_th]))==0 and len(listchar)>0 and listchar[-1] !=-1:
                if occ >thr:
                    if last_max >= conf:
                        listchar.append(lastone)
                    else:
                        listchar.append(-2)
                occ = 0
                listchar.append(-1)
                lastone = -1
                last_max = 0
        if (np.where(matrix[:,z] == maxs[z])[0].shape[0]==1):
            a = (np.where(matrix[:,z] == maxs[z])[0][0])
            if a!=lastone or last_max!= maxs[z]:
                if occ > thr:
                    if last_max >= conf:
                        listchar.append(lastone)
                    else:
                        listchar.append(-2)
                occ = 0
                lastone = a
                last_max = maxs[z]
            else:
                occ = occ +1
    if occ > thr:
        if last_max >= conf:
            listchar.append(lastone)
        else:
            listchar.append(-2)
    return(listchar)

def draw_and_read():
    
    model.eval()

    matrices = []

    shots = 5
    
    stop=0
    i=0
    for t in  (list_lines[:]):
        img1 = Image.open('eval_borg/lines/'+t).convert("RGB")
        img1 = Fsupp.to_tensor(img1)

        probs, matrix = drawprobs(img1,shots,1,len(alphabet_fo))
        matrices.append(matrix)
    return(matrices)

def zid_read(matrices):
    conf = 0.3
    results = []
    for matrix in matrices:
        l_ch = read_sp_char(matrix,22)

        try:
            if l_ch[0]==-1:
                l_ch.pop(0)
            if l_ch[-1]==-1:
                l_ch.pop()
        except:
            continue
        results.append(l_ch)
    return results




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
    text= text.split('\n')[0]
    text = text.split(' ')
    for c in text:
        if c not in alphabet_fo: #### borg
            continue
            res.append(-3)   #### if you want to ignore out of vocab symbols make it continue
        else:
            res.append(alphabet_fo.index(c))
    return (res)


def asciitochar(a):
    string = ''
    for ch in a:
        string = string+chr(97+ch)
    return(string)

def get_error_rate(gt,pred):
    qo = 0
    acc = 0
    word_acc=0
    all_symb = 0
    missing_symbs = 0
    for gt_line,pred_line in zip(gt,pred):
        gt_text = asciitochar(gt_line)
        pred_text = asciitochar(pred_line)
        qo = qo+1
        cer = (editdistance.eval(gt_text, pred_text)-pred_line.count(-2))/len(gt_text)
        all_symb = all_symb+len(gt_text)
        missing_symbs = missing_symbs + pred_line.count(-2)
        acc = acc + cer
        if cer ==0:
            word_acc = word_acc+1
    if qo==0:
        return (1,word_acc)
    else:
        return (acc/qo),word_acc


                
def clean_labels(X_lab2):
    to_delete2 = set()
    for key in X_lab2.keys():
        if X_lab2[key]['bboxes'][0]['class']==1:
            k1 = key.split('.jpg')[0]

            for key2 in X_lab2.keys():
                if X_lab2[key2]['bboxes'][0]['class']==1:
        
                    k2 = key2.split('.jpg')[0]
                    if (k1 == k2) and X_lab2[key]['class']!= X_lab2[key2]['class']:
                        
                        x1b1 = X_lab2[key]['bboxes'][0]['x1']
                        x2b1 = X_lab2[key]['bboxes'][0]['x2']
                        
                        x1b2 = X_lab2[key2]['bboxes'][0]['x1']
                        x2b2 = X_lab2[key2]['bboxes'][0]['x2']
                        
                        if(same_box(x1b1,x2b1,x1b2,x2b2)):
                            if X_lab2[key]['bboxes'][0]['score'] < X_lab2[key2]['bboxes'][0]['score']:
                                to_delete2.add(key)
                            else:
                                to_delete2.add(key2)

    for d_key in to_delete2:
        del X_lab2[d_key]
    return X_lab2             

pages = cipher+'_'

best_cer = 1
batch_s = 5
best_epoch = 0
start_eval = 7



shots_number = 5
shots_path = 'alphabet/'+cipher
alphabet_fo = os.listdir(shots_path)






model.load_state_dict(torch.load('weights/synthetic_'+cipher+'.pth'))


select_new_labels(L,0.4,0.2,starting = True,the_epoch=0)
L = readposneg()
# show_new(L,epoch)
dataset_lab, data_loader = load_data(batch_s=batch_s, shots_number=shots_number,root = 'few5', L=L)



start_eval = 7  ############
val_data_path = "data_validation"
val_lines_path = val_data_path+'/lines/'
val_text_path  = val_data_path+'/gt/'

# model.load_state_dict(torch.load('model/model_epoch_'+str(30)+'.pth'))

for epoch in range(0, 1500):
    
    start_eval-=1
    
    if  start_eval <= 0: # and epoch % 3 ==0:
        if start_eval==0:
            curr_cer = 1
        list_lines =  os.listdir(val_lines_path+cipher)[:20]
        
        results = draw_and_read()
        gt = get_gt()
        cer = get_error_rate(gt,zid_read(results))[0]
        

        print('CER: ',cer)
        print('Last best CER:', best_cer)
        print('Best epoch:', best_epoch)
        
        


        if cer<best_cer:
            best_cer = cer
            best_epoch = epoch
        if cer<curr_cer:
            print('Model saved at cer:' , cer)
            curr_cer = cer
            torch.save(model.state_dict(), 'weights/progressive_'+cipher+'.pth')
    if  epoch>0 and epoch % 15==0:#######    
        
        
        model.load_state_dict(torch.load('weights/progressive_'+cipher+'.pth'))
        
        select_new_labels(L,0.4,0.2,starting = epoch==0,the_epoch=epoch)
        
        L = readposneg()

        # show_new(L,epoch)
        dataset_lab, data_loader = load_data(batch_s=batch_s, shots_number=shots_number,root = 'few5', L=L)
            
        
        model.load_state_dict(torch.load('weights/synthetic_'+cipher+'.pth'))

        start_eval = 7  ##########""

    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=int(len(dataset_lab)/batch_s/5))
