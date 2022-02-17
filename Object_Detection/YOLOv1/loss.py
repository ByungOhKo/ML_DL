import torch
import torch.nn as nn
from utils import intersection_over_union as iou

class YoloLoss(nn.Module):
    # YOLO의 perdiction의 shape은 (N, S * S * (C + B * 5)) 가 된다.
    # 논문에 주어진 parameter와 맞추면 (N, 1470)이 된다.
    # 이를 reshape을 통해 (N, S, S, C + B * 5) = (N, 7, 7, 30)으로 변환한다.
    
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
    
    def forward(self, pred, target):
        # predictions.shape = (BATCH_SIZE, S, S, C + B * 5)
        pred = pred.reshape(-1, self.S, self.S, self.C + self.B * 5)
        
        iou_all = [iou(pred[..., self.C + 1 + 5*i : self.C + 1 + 5*(i+1)]) for i in range(self.B)]
        
        return iou_all