import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()

        self.channels = 32
        layers = [nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )]
        for _ in range(6):
            layers.append(nn.Sequential(
                nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True)
            ))
        
        layers.append(nn.Sequential(
            nn.Conv2d(64,30,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU(True)
        ))

        layers.append(nn.AdaptiveAvgPool2d(output_size = (1,1)))
        self.layers = nn.Sequential(*layers)
        self.linear = nn.Linear(30,30)

    def forward(self,image_with_message):
        x = self.layers(image_with_message)
        x.squeeze_(3).squeeze_(2)
        x = self.linear(x)
        return x
