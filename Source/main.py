# 가장 기본 라이브러리
import os
import time
import torch
import torchvision
import glob
import pandas as pd 
import numpy as np 
from torchvision import datasets, transforms
# 내가 만든 라이브러리
import utils
from teacher.Basic_model import Basic
from model.Student_model import Student
from Noiser.Noiser import Noiser
import random
def main():
    # gpu 설정
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed=1
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # 하이퍼 파라미터 선언부
    batch_size = 12
    epochs = 400
    message_size = 30
    
    height = 128
    weight = 128

    # 데이터 및 모델 선언부
    data_path = '../data/COCO/'
    image_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop((height,weight), pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ]),
        'test': transforms.Compose([
            transforms.RandomCrop((height,weight),pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
    }

    train_images = datasets.ImageFolder(data_path+'train/',image_transforms['train'])
    train_loader = torch.utils.data.DataLoader(train_images,batch_size = batch_size ,num_workers=2,shuffle=True)

    valid_images = datasets.ImageFolder(data_path+'val/',image_transforms['test'])
    valid_loader = torch.utils.data.DataLoader(valid_images,batch_size = batch_size ,num_workers=2,shuffle=True)   

    sub_images = datasets.ImageFolder(data_path+'train/',image_transforms['test'])
    sub_loader = torch.utils.data.DataLoader(sub_images,batch_size = batch_size ,num_workers=2,shuffle=True)

    noiser = Noiser(device,'teacher')
    teacher_model = Basic(device,noiser)
    #student_noiser = Noiser(device,'student')
    #student_model = Student(device,student_noiser)

    # 학습파트
    # - 학습하면서 csv 파일에 이미지 - 텍스트 쌍 저장할 것
    # - 10 epoch 마다 모델 저장
    df = pd.DataFrame()
    print('='*30)
    t_best_loss = 100
    s_best_loss = 100
    
    for epoch in range(epochs):
        epoch_start = time.time()
        t_loss = []
        s_loss = []
        #학습데이터셋
        for (image, _), (sub_image, _) in zip(train_loader,sub_loader):
            image = image.to(device)
            sub_image = sub_image.to(device)
            #message = torch.randint(0,2,(image.shape[0],message_size),dtype=torch.bool).to(device)
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_size))).to(device)
        
            
            t_losses, t_encoded_image, t_decoded_message = teacher_model.train_on_batch([image,message,sub_image],epoch)
           # s_losses, s_decoded_message = student_model.train_on_batch([t_encoded_image.detach(),message,sub_image,t_decoded_message[0].detach()],epoch) 

        #학습 시간 출력
        epoch_end = time.time()
        print('training time : ' , epoch_end - epoch_start)
        
        #검증데이터셋
        for image, _ in valid_loader:
            image = image.to(device)
            #message = torch.randint(0,2,(image.shape[0],message_size),dtype=torch.bool).to(device)
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_size))).to(device)
            t_losses, t_encoded_image, t_decoded_message = teacher_model.validate_on_batch([image,message,0],epoch) 
            #s_losses, s_decoded_message = student_model.valid_on_batch([t_encoded_image,message,0],epoch)
            t_loss.append(t_losses['decoder_mse'])
            #s_loss.append(s_losses['loss'])
        #로스 출력
        print(epoch)
        print('teacher : ', t_losses)
        #print('student : ', s_losses, sum(s_loss)/len(s_loss))

        utils.save_checkpoint(teacher_model,epoch,'./result/checkpoints/')            
        #utils.save_student(student_model,epoch,'./result/checkpoints/')
            



if __name__ == '__main__':
    main()
