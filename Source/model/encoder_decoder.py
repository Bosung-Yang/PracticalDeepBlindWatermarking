import model.decoder
import model.encoder
import torch.nn as nn

class EncoderDecoder(nn.Module):
    def __init__(self,noiser):
        super(EncoderDecoder,self).__init__()
        self.encoder = model.encoder.Encoder()
        self.decoder = model.decoder.Decoder()
        self.noiser = noiser

    def forward(self,image,message,sub_image, epoch, mode):
        encoded_image = self.encoder(image,message)
        if mode == 'train':
            noised_image = self.noiser(encoded_image,sub_image,epoch)
            decoded_massage = self.decoder(encoded_image)
            return encoded_image, decoded_massage, noised_image
        elif mode == 'valid':
            decoded_massage = self.decoder(encoded_image)
            return encoded_image, decoded_massage, 0
