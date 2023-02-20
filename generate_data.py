import random
import matplotlib.pyplot  as plt
import os
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
import cv2 
import sys


def crop_im(image):
    a1=0
    a2=0
    b1=0
    b2=0
    for i in range (image.shape[1]):
        if (sum(image[:,i])<104.0 and a1==0):
            a1=i
    for i in range (image.shape[1]):
        if (sum(image[i,:])<104.0 and b1==0):
            b1=i
        
    for j in range (image.shape[1]-1,0,-1):
        if (sum(image[:,j])<104.0 and a2==0):
            a2=j
    
    for j in range (image.shape[1]-1,0,-1):
        if (sum(image[j,:])<104.0 and b2==0):
            b2=j
       
    return image[:,a1:a2]





def merge(a,b,r):
    

    imagemix = np.ones((a.shape[0],a.shape[1]+b.shape[1]-r))
    imagemix[:,:a.shape[1]]=a
    
    for i in range (a.shape[0]):
        for j in range (a.shape[1]-r,a.shape[1]+b.shape[1]-r):
            if (j<a.shape[1]):
                imagemix[i,j] = a[i,j]*b[i][j-(a.shape[1]-r)]
            else:
                imagemix[i,j] = b[i][j-(a.shape[1]-r)]
    return imagemix




def get_boundary(image):
    max_p = np.max(image)
    a=0
    b=image.shape[0]
    for i in range (image.shape[1]):
        if (sum(image[:,i])<max_p*image.shape[0]-2*max_p and a==0):
            a=i
        
    for j in range (image.shape[1]-1,0,-1):
        if (sum(image[:,j])<max_p*image.shape[0]-2*max_p and b==image.shape[0]):
            b=j

    
    c=0
    d=image.shape[1]
    for i in range (image.shape[0]):
        if (sum(image[i,:])<max_p*image.shape[1]-2*max_p and c==0):
            c=i
        
    for j in range (image.shape[0]-1,0,-1):
        if (sum(image[j,:])<max_p*image.shape[1]-2*max_p and d==image.shape[1]):
            d=j
    return a,b,c,d



def inverse(a):
    return np.max(a)-a



def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

def preprocess(a):
    a = inverse(a)
    angle = random.randint(-5,5)
    scale = random.randint(8,11) /10
    b= rotate (a,angle, scale=scale)
    return (b<0.2*np.max(a)) * np.max(a)

def crop_and_pad(im):
    _,_,h1,h2 = get_boundary(im)
    
    cat1 = random.randint(0,7)
    cat2 = random.randint(0,7)
    
    im = im[h1+cat1:h2-cat2,:]
    
    randompad = np.ones((105,im.shape[1]))
    v = 105 - im.shape[0]
    s  = random.randint(int(v/4),int(v/1.1))
    randompad[s:s+im.shape[0],:] = im
#     im = cv2.resize(im, (im.shape[1],105))
    return randompad

def add_noise(im,t):
    a = cv2.imread((path+'/line'+str(random.randint(1,t))+'.png'),0)
    if a.shape[1]>im.shape[1]:
        a = a[:,:im.shape[1]]
    _,_,h1,h2 = get_boundary(a)
    a = a[h1:h2,:]
    
    noise1 = random.randint(10,20)
    noise2 = random.randint(10,20)
    
    im = inverse(im)
    a = inverse(a)
    if random.randint(1,1):
        im[:noise1,:a.shape[1]] = im[:noise1,:a.shape[1]] + a[a.shape[0]-noise1:,:] 
    
    if random.randint(1,1):
        im[im.shape[0]-noise2:,:a.shape[1]] = im[im.shape[0]-noise2:,:a.shape[1]] + a[:noise2,:] 
    
    im[im>255] = 255
    return inverse(im)



def annotate():

    train_df = pd.read_csv('./csv_file.csv')

    f= open("few5/annotation/synthetic.txt","a")
    for idx, row in train_df.iterrows():
        
        path0  = row['FileName']
        pathr = path0.split('.')[0]+'.png'
        path0 = path0.split('.')[0]+'.jpg'
        img = cv2.imread(( '' + pathr))
        height, width = img.shape[:2]
        x1 = int(row['XMin'] )# * width)
        x2 = int(row['XMax'] )#* width)
        y1 = int(row['YMin'] )#* height)
        y2 = int(row['YMax'] )#* height)

        # google_colab_file_path = 'few5'
        fileName = os.path.join( path0)
        className = row['Class']
        f.write(fileName + ',' + str(x1) + ',' + str(y1) + ',' 
                    + str(x2) + ',' + str(y2) + ',' + str(className) + '\n')
    f.close()



def preprocess2(a):
    kernel = np.ones((random.randint(1,3),random.randint(1,2)),np.uint8)
    a = cv2.dilate(a,kernel,iterations = 1)
    return a



def merge(a,b):
    
    first_b,_,_,_ = get_boundary(b)

    a = inverse(a)
    b = inverse(b)
    
    imagemix = np.zeros((a.shape[0],a.shape[1]+b.shape[1]+30))
    imagemix[:,:a.shape[1]]=a[:,:a.shape[1]]
    keep = True
    
    c = a.shape[1]
    
    while keep:
        if (np.max(imagemix[:,c:c+b.shape[1]]+b)>1) or c <=10:
            keep = False
        c=c-1
            
    
    c= c+random.randint(0,30)
    imagemix[:,c:c+b.shape[1]]=imagemix[:,c:c+b.shape[1]]+b
    
    cut = True
    cutting = 20
    while cut:
        if np.sum(imagemix[:,imagemix.shape[1]-cutting:])==0:
            imagemix = imagemix[:,:imagemix.shape[1]-cutting]
            
            cutting = cutting+20
        else:
            cut = False
    
    return inverse(imagemix)



def create_data(new,path,number,s_path_l = ['synthetic']):
    try:
        os.makedirs('few5/annotation')
    except:
        if (os.path.exists("few5/annotation/synthetic.txt") and new):
            os.remove("few5/annotation/synthetic.txt")
    line=1
    train_df = pd.DataFrame(columns=['FileName', 'XMin', 'XMax', 'YMin', 'YMax', 'Class'])
    for  l in tqdm(range(number)):
        s_path = random.choice(s_path_l)

        r_folder = random.choice(os.listdir(s_path))
        r_element= s_path+'/'+r_folder+'/'+random.choice(os.listdir(s_path+'/'+r_folder))
        
        a =  cv2.imread(r_element,0)#[:,:,0]
        a = preprocess(a)
        b1,b2,_,_ = get_boundary(a)
        
        train_df=train_df.append({'FileName': path+"/line"+str(line)+".png",
                                    'XMin': b1,'XMax': b2,'YMin': 0,'YMax': 105,
                                    'Class': r_folder},ignore_index=True)
         #+ 3
        for  i in range (random.randint(35,50)):

            s_path = random.choice(s_path_l)

            r_folder = random.choice(os.listdir(s_path))
            r_element= s_path+'/'+r_folder+'/'+random.choice(os.listdir(s_path+'/'+r_folder))
            
            b =  cv2.imread(r_element,0)#[:,:,0]
            b = preprocess (b)
            
            a= merge(a,b)
            
            a1,a2,_,_ = get_boundary(a)
            b1,b2,_,_ = get_boundary(b)
            
            
            train_df=train_df.append({'FileName': path+"/line"+str(line)+".png",
                                        'XMin': a2-(b2-b1),'XMax': a2,'YMin': 0,
                                        'YMax': 105,'Class': r_folder},ignore_index=True)
        # a=preprocess(a)
        plt.imsave(path+'/line'+str(line)+'.png',a,cmap='gray')
        
        line+=1
        
        train_df.to_csv('./csv_file.csv')
        train_df = pd.DataFrame(columns=['FileName', 'XMin', 'XMax', 'YMin', 'YMax', 'Class'])
        annotate()


path = 'few5/synthetic'
if not os.path.exists(path):
    os.makedirs(path)

tr_lines = 250
new = True


slecing_path = ['alphabet/vatican']
create_data(new,path,tr_lines,s_path_l = slecing_path)

for i in tqdm(range(1,tr_lines+1)):
    a =  crop_and_pad(cv2.imread((path+'/line'+str(i)+'.png'),0))
    a = add_noise(a,tr_lines)
    plt.imsave(path+'/line'+str(i)+'.jpg',a,cmap='gray')
os.system("rm "+path+"/*.png")
