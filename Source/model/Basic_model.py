import numpy as np
import torch
import torch.nn as nn
import model.discriminator
import model.encoder_decoder

class Basic:
    def __init__(self, device:torch.device, noiser):

        super(Basic,self).__init__()
        self.noiser = noiser
        self.ED = model.encoder_decoder.EncoderDecoder(self.noiser).to(device)
        self.DIS = model.discriminator.Discriminator().to(device)
        self.optim_ED = torch.optim.Adam(self.ED.parameters())
        self.optim_DIS = torch.optim.Adam(self.DIS.parameters())

        self.device = device
        
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device)
        self.mse_loss = nn.MSELoss().to(device)

        self.cover_label = 1
        self.encoded_label = 0

    def train_on_batch(self, batch:list, epoch):
        images, message, sub_image = batch

        batch_size = images.shape[0]

        #학습 모드로 변환
        self.ED.train()
        self.DIS.train()

        with torch.enable_grad():
            
            self.optim_ED.zero_grad()
            self.optim_DIS.zero_grad()
            #판별자 - 진짜 이미지 학습 -> 1로 예측하도록
            dis_real_labels = torch.full((batch_size,1),self.cover_label,device=self.device).float()
            predict_real = self.DIS(images)
            dis_loss_real = self.bce_with_logits_loss(predict_real, dis_real_labels)
            dis_loss_real.backward()
            

            #판별자 - 가짜 이미지 학습 -> 0으로 예측하도록
            dis_fake_labels = torch.full((batch_size,1),self.encoded_label,device = self.device).float()
            encoded_image , decoded_message, noised_image = self.ED(images,message,sub_image,epoch,'train')
            predict_fake = self.DIS(encoded_image.detach())
            dis_loss_fake = self.bce_with_logits_loss(predict_fake, dis_fake_labels)
            dis_loss_fake.backward()

            #판별자 옵티마이저
            self.optim_DIS.step()

            #생성자 학습
            gen_target_label = torch.full((batch_size,1),self.cover_label,device=self.device).float()
            predict_encoded = self.DIS(encoded_image)
            #생성자 - 판별자 속이기 위한 로스 
            gen_loss_adv = self.bce_with_logits_loss(predict_encoded, gen_target_label)
            #생성자 - 이미지 퀄리티를 위한 로스
            gen_loss_enc = self.mse_loss(encoded_image, images)
            #생성자 - 디코딩 정확도를 위한 로스
            gen_loss_dec = self.mse_loss(decoded_message, message)
            

            gen_loss = (0.01*gen_loss_adv +  0.8*gen_loss_enc + 1*gen_loss_dec )
            gen_loss.backward()
            self.optim_ED.step()

        losses = {
            'loss':gen_loss.item(),
            'encoder_mse' : gen_loss_enc.item(),
            'decoder_mse' : gen_loss_dec.item(),
        }

        return losses, encoded_image, decoded_message

    def validate_on_batch(self,batch:list,epoch):
        images,message,sub_image = batch
        batch_size = images.shape[0]

        self.ED.eval()
        self.DIS.eval()

        with torch.no_grad():

            #판별자 - 진짜 이미지 검증 -> 1로 예측하도록
            dis_real_labels = torch.full((batch_size,1),self.cover_label,device=self.device).float()
            predict_real = self.DIS(images)
            dis_loss_real = self.bce_with_logits_loss(predict_real, dis_real_labels)

            #판별자 - 가짜 이미지 검증 -> 0으로 예측하도록
            dis_fake_labels = torch.full((batch_size,1),self.encoded_label,device = self.device).float()
            encoded_image , decoded_message, noised_image = self.ED(images,message,sub_image,epoch,'valid')
            predict_fake = self.DIS(encoded_image)
            dis_loss_fake = self.bce_with_logits_loss(predict_fake, dis_fake_labels)
            
            #생성자 학습
            gen_target_label = torch.full((batch_size,1),self.cover_label,device=self.device).float()
            predict_encoded = self.DIS(encoded_image)
            #생성자 - 판별자 속이기 위한 로스 
            gen_loss_adv = self.bce_with_logits_loss(predict_encoded, gen_target_label)
            #생성자 - 이미지 퀄리티를 위한 로스
            gen_loss_enc = self.mse_loss(encoded_image, images)
            #생성자 - 디코딩 정확도를 위한 로스
            gen_loss_dec = self.mse_loss(decoded_message, message)

        decoded_rounded = decoded_message.detach().cpu().numpy().round().clip(0,1)
        bitwise_err = np.sum(np.abs(decoded_rounded - message.detach().cpu().numpy())) / (batch_size * message.shape[1])
        losses = {
            'encoder_mse' : gen_loss_enc.item(),
            'decoder_mse' : gen_loss_dec.item(),
            'bit_err ' : bitwise_err,
        }

        return losses, encoded_image, decoded_message
    
    def decode_from_image(self,image):
        self.ED.eval()
        with torch.no_grad():
            decoded_message = self.ED.decoder(image)
        return decoded_message       
            







