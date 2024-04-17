import torch
from torch import nn
from MultiheadWDCNN import multiheadWDCNN


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.multihead_WDCNN = multiheadWDCNN()

    def forward(self, x):
        #   Input sample pair:
        x1, x2 = x  # x1:[BatchSize, 1, 6284]  x2:[BatchSize, 1, 6284]

        #   Feature extraction:
        y1 = self.multihead_WDCNN(x1)  # y1:[BatchSize, 100]
        y2 = self.multihead_WDCNN(x2)  # y2:[BatchSize, 100]

        #   Similarity measurement (Cosine Similarity):
        similarity = torch.cosine_similarity(y1, y2, dim=1)   # similarity:[BatchSize]

        #   Output similarity:
        return torch.sigmoid(similarity)
