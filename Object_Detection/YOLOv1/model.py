from platform import architecture
import torch
import torch.nn as nn


architecture_config = [
    # YOLO 논문의 (Figure 3: The Architecture)의 Conv, Maxpool layers를 선언
    # tuple: (kernel_size, channels, stride, padding)
    # str: Maxpool layer
    # list: [tuple1, tuple2, iteration]
    (7, 64, 2, 3),
    "Maxpool",
    (3, 192, 1, 1),
    "Maxpool",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "Maxpool",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "Maxpool",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

# 논문에 적힌 내용 
# We use a linear activation function for the final layer and
# all other layers use the following leaky rectified linear activation

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.leakyrelu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        return self.leakyrelu(self.conv(x))
    
class Yolov1(nn.Module):
    def __init__(self, architecture_config, in_channels, grid_size, num_boxes, num_classes):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(grid_size, num_boxes, num_classes)
        
    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))
        
    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        
        for cfg in architecture:
            if type(cfg) == tuple:
                layers += [CNNBlock(in_channels, cfg[1], kernel_size=cfg[0], stride=cfg[2], padding=cfg[3])]
                in_channels = cfg[1]
                
            elif type(cfg) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                
            elif type(cfg) == list:
                iter = cfg[-1]
                for _ in range(iter):
                    for cfg_in in cfg[:-1]:
                        layers += [CNNBlock(in_channels, cfg_in[1], kernel_size=cfg_in[0], stride=cfg_in[2], padding=cfg_in[3])]
                        in_channels = cfg_in[1]
                    
        return nn.Sequential(*layers)
    
    def _create_fcs(self, grid_size, num_boxes, num_classes):
        S, B, C = grid_size, num_boxes, num_classes

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, S * S * (C + B * 5)),
        )
        
def test():
    model = Yolov1(architecture_config, 3, 7, 2, 20)
    x = torch.randn((2, 3, 448, 448))
    print(model(x).shape)