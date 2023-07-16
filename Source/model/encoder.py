import torch
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder,self).__init__()
        self.H = 128 # height
        self.W = 128 # width
        self.nc = 3 # number of channel
        self.conv_channel = 64

        self.input_image = nn.Sequential(
            nn.Conv2d(30+3,self.conv_channel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(self.conv_channel),
            nn.ReLU(True),

            nn.Conv2d(self.conv_channel,self.conv_channel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(self.conv_channel),
            nn.ReLU(True),

            nn.Conv2d(self.conv_channel,self.conv_channel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(self.conv_channel),
            nn.ReLU(True),

            nn.Conv2d(self.conv_channel,self.conv_channel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(self.conv_channel),
            nn.ReLU(True),
            )

        self.after_concat = nn.Sequential(
            nn.Conv2d(self.conv_channel,self.conv_channel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(self.conv_channel),
            nn.ReLU(True),
        )

        self.output_layer = nn.Conv2d(self.conv_channel,3,kernel_size=1)
        self.before_output = nn.Conv2d(self.conv_channel+3,64,kernel_size=3,stride=1,padding=1)


    def forward(self,image,message):
        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)
        expanded_message = expanded_message.expand(-1,-1,self.H,self.W)
        concat = torch.cat([expanded_message,image],dim=1)
        concat = self.input_image(concat)
        encoded_image = self.after_concat(concat)
        encoded_image = torch.cat([encoded_image,image],dim=1)
        encoded_image = self.before_output(encoded_image)
        encoded_image = self.output_layer(encoded_image)
        return encoded_image
    
