import torch
from tqdm import tqdm
from torchvision.transforms import functional as Fsupp
import os
import numpy as np
from configs import getOptions
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import editdistance
import random


options = getOptions().parse()

alphabet_path = options.alphabet
resizing = options.resize

threshold = options.thresh
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def asciitochar(a):
    string = ''
    for ch in a:
        string = string+chr(50+ch)
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
    if qo>0:
        return (acc/qo),word_acc
    else:
        return 1,word_acc


def drawprobs(model, cipher, img1,shots,st_ch,en_ch):
    mat_size  = 100
    img2_size = 105
    if resizing:
        img2_size = 128
    
    image_hline = Image.new('RGB', (img1.size()[2]+5+mat_size, 5), (0, 0, 255))
    image1 = Image.fromarray(img1.mul(255).permute(1, 2, 0).byte().numpy())
    image_f = Image.new('RGB', (mat_size, img2_size), (255, 255, 255))
    image_vline = Image.new('RGB', (5, img2_size), (0, 0, 255))
    imgs_comb = np.hstack( (image_f,image_vline,image1) )

    thresh = threshold
    
    font = ImageFont.truetype("src/arial.ttf", 25)
    
    Pro_matrix = np.zeros((en_ch-st_ch +1,img1.size()[2]))
    last_max = 0
    p_c = 0
    for symbol in os.listdir(alphabet_path+'/'+cipher):
        Matrix  = torch.zeros((3,mat_size,img1.size()[2]))
        
        i_symbs =  os.listdir(alphabet_path+'/'+cipher+'/'+symbol)
        random.shuffle(i_symbs)
        i_symbs = i_symbs[:shots] 

        for symb in i_symbs:

            try:
                img2 = Image.open(alphabet_path+'/'+cipher+'/'+cipher+'/'+symbol+'/'+symb.split('.png')[0]+'.jpg').convert("RGB")
            except:
                img2 = Image.open(alphabet_path+'/'+cipher+'/'+symbol+'/'+symb.split('.jpg')[0]+'.jpg').convert("RGB")
            if resizing:
                img2 = img2.resize((img2_size,img2_size))
            img2 = Fsupp.to_tensor(img2)

            with torch.no_grad():
                preds = model([img1.to(device)],[img2.to(device)])
            preds = preds[0]
            
            
        
            for  box,lab in zip (preds['boxes'],range(preds['scores'].size()[0])):
                if (preds['scores'][lab].item()>thresh):
                    Mat = torch.zeros((3,mat_size,int(box[2].item())-int(box[0].item()))) + preds['scores'][lab].item()
                    Matrix[:,:,int(box[0].item()):int(box[2].item()) ] = torch.max(Mat,Matrix[:,:,int(box[0].item()):int(box[2].item()) ]) 

                    Pmat = np.zeros((1,int(box[2].item())-int(box[0].item()))) + preds['scores'][lab].item() 
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
        

    imgs_comb = Image.fromarray( imgs_comb)
    return imgs_comb, Pro_matrix




# read the spaces
def read_sp_char(matrix,thr,conf = 0.3):
    maxs = matrix.max(axis=0)
    listchar = []
    list_boxes=[]
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





# dont read the spaces
def read_char(matrix,thr,conf = 0.3):
    maxs = matrix.max(axis=0)
    
    listchar = []
    list_boxes=[]
    
    end_box=False
    occ = 0
    lastone=0
    last_max = 0

    
    
    for z in range (matrix.shape[1]):
        if (np.where(matrix[:,z] == maxs[z])[0].shape[0]==1):
            a = (np.where(matrix[:,z] == maxs[z])[0][0])
            if a!=lastone or last_max!= maxs[z]:
                if occ > thr:

                    list_boxes.append(z-occ)
                    end_box = True
                    if last_max >= conf:
                        listchar.append(lastone)
                    else:
                        listchar.append(-2)
                    occ = 0
                lastone = a
                last_max = maxs[z]
                if end_box:
                    list_boxes.append(z)
                    end_box = False

            else:
                occ = occ +1
    if occ > thr:
        last_row = matrix[lastone]
        i_c = matrix.shape[1]-1
        last_v = last_row[i_c]

        while (last_row[i_c]==last_v):
            i_c -=1
        l_b_b = i_c
        last_v = last_row[i_c]

        while (last_row[i_c]==last_v):
            i_c -=1
        l_b_a = i_c
        


        list_boxes.append(l_b_a)
        list_boxes.append(l_b_b)
        
        
        if last_max >= conf:
            listchar.append(lastone)
        else:
            listchar.append(-2)
        
    return(listchar,list_boxes)



def draw_and_read(model,list_lines,lines_path,cipher,shots_number):
    
    model.eval()
    matrices = []
    
    stop=0
    i=0
    for t in  tqdm (list_lines[:]):
        img1 = Image.open(lines_path+'/'+cipher+'/'+t).convert("RGB")
        if resizing:
            img1 = img1.resize((2048,128))
        img1 = Fsupp.to_tensor(img1)

        _, matrix = drawprobs(model,cipher,img1,shots_number,1,len(os.listdir(alphabet_path+'/'+cipher)))
        matrices.append(matrix)
    return(matrices)

def zid_read(matrices,read_space=False):
    thresh = threshold
    results = []
    box_results = []
    for matrix in matrices:

        if read_space:
            l_ch,l_boxes = read_sp_char(matrix,22,conf= thresh)
        else:
            l_ch,l_boxes = read_char(matrix,22,conf= thresh)

        try:
            if l_ch[0]==-1:
                l_ch.pop(0)
            if l_ch[-1]==-1:
                l_ch.pop()
        except:
            continue
        results.append(l_ch)
        box_results.append(l_boxes)
    return results, box_results

def inttosymbs(preds,cipher):
    alphabet_symbs = os.listdir(alphabet_path+'/'+cipher) 
    pred_lines = []
    
    for pr in preds:
        p_line=''
        for i in range (len(pr)):
            if pr[i]==-1:
                p_line += ' '
            else:
                if pr[i]==-2:
                    p_line += '*' + ' '
                else:
                    p_line += alphabet_symbs[pr[i]] + ' '
        pred_lines.append(p_line[:-1])
    return pred_lines
