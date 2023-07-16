import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        
        layers = [nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        ),nn.Sequential(
                    nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True)
        ),nn.Sequential(
                    nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True)
        ),nn.Sequential(
                    nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True)
        )
        ]

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1,1)))

        self.before_linear = nn.Sequential(*layers)
        self.linear = nn.Linear(64,1)

    def forward(self,image):
        X = self.before_linear(image)
        X.squeeze_(3).squeeze_(2)
        X = self.linear(X)
        return X
