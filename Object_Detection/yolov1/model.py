import torch
import torch.nn as nn

arch_cfg = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]
# tuple: (kernel size, number of filters of output, stride, padding)
# str:   max-pooling 2x2 with stride = 2
# list:  [tuple, tuple, repeat times]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
        # Batch normalization은 애초에 gamma * normalized(x) + bias 로 적용되기 때문에
        # conv layer에서 conv(x) + bias에서 bias의 존재는 의미가 없다.
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)
        return x
    

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, architecture_config=arch_cfg, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_darknet(self.architecture)
        self.fcs = self._create_fcs(**kwargs)
        
    def forward(self, x):
        x = self.darknet(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fcs(x)
        return x
    
    def _create_darknet(self, architecture):
        layers = []
        in_channels = self.in_channels
        
        for x in architecture:
            if type(x) == tuple:
                # conv layer
                layers += [
                    CNNBlock(in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3])
                    ]
                in_channels = x[1]
            
            elif type(x) == str:
                # max pooling layer
                layers += [
                    nn.MaxPool2d(kernel_size=2, stride=2)
                ]
            
            elif type(x) == list:
                # conv layer with repeat
                for _ in range(x[-1]):
                    for conv in x[:-1]:
                        layers += [
                            CNNBlock(in_channels, conv[1], kernel_size=conv[0], stride=conv[2], padding=conv[3])
                            ]
                        in_channels = conv[1]                     
    
        return nn.Sequential(*layers)
    
    def _create_fcs(self, grid_size, num_boxes, num_classes):
        S, B, C = grid_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(S*S*1024, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S*S*(B*5+C)),
        )