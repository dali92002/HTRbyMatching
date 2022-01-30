from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import numpy as np
import os
import PIL
import torch








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
        random.shuffle(choices)
        for j in range (shots):
            num_s = len(os.listdir('eval_borg/borg_symbs/'+alphabet_fo[i]))
            
            img2 = Image.open('eval_borg/borg_symbs/'+alphabet_fo[i]+'/'+str(choices[0])+'.jpg').convert("RGB")
            img2 = Fsupp.to_tensor(img2)
            choices.pop(0)
            with torch.no_grad():
                _ = model([img1.to(device)],[img2.to(device)])
            _ = _[0]
            
            
        # i = 0
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
#         drr = 5
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
