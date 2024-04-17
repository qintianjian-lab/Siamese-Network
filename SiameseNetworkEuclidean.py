import torch
from torch import nn
from MultiheadWDCNN import multiheadWDCNN


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.multihead_WDCNN = multiheadWDCNN()
        self.FC = torch.nn.Linear(100, 1)

    def forward(self, x):
        #   Input sample pair:
        x1, x2 = x  # x1:[BatchSize, 1, 6284]  x2:[BatchSize, 1, 6284]

        #   Feature extraction:
        y1 = self.multihead_WDCNN(x1)  # y1:[BatchSize, 100]
        y2 = self.multihead_WDCNN(x2)  # y2:[BatchSize, 100]

        #   Similarity measurement (Euclidean Distance):
        c = (y1 - y2) ** 2  # c:[BatchSize, 100]
        similarity = self.FC(c)  # similarity:[BatchSize, 1]
        similarity = similarity.squeeze()  # similarity:[BatchSize]

        #   Output similarity:
        return torch.sigmoid(similarity)
