import torch
from torch import nn


class multiheadWDCNN(nn.Module):

    def __init__(self, in_channel=1, out_channel=2):
        super(multiheadWDCNN, self).__init__()

        self.layer1_1 = nn.Sequential(
            # [BatchSize, 1, 6284]
            nn.Conv1d(in_channels=in_channel, out_channels=16, kernel_size=(32,), stride=(8,), padding=2),
            # [BatchSize, 16, 783]
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=(2,), stride=(2,), padding=1)
            # [BatchSize, 16, 392]
        )

        self.layer1_2 = nn.Sequential(
            # [BatchSize, 1, 6284]
            nn.Conv1d(in_channels=in_channel, out_channels=16, kernel_size=(64,), stride=(16,), padding=2),
            # [BatchSize, 16, 390]
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=(2,), stride=(2,))
            # [BatchSize, 16, 195]
        )

        self.layer1_3 = nn.Sequential(
            # [BatchSize, 1, 6284]
            nn.Conv1d(in_channels=in_channel, out_channels=16, kernel_size=(128,), stride=(32,), padding=10),
            # [BatchSize, 16, 194]
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=(2,), stride=(2,))
            # [BatchSize, 16, 97]
        )

        self.layer2 = nn.Sequential(
            # [BatchSize, 16, 684]
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=(3,), stride=(1,), padding=1),
            # [BatchSize, 32, 684]
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=(2,), stride=(2,))
            # [BatchSize, 32, 342]
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=(3,), stride=(1,), padding=1),
            # [BatchSize, 64, 342]
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=(2,), stride=(2,))
            # [BatchSize, 64, 171]
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=(3,), stride=(1,), padding=1),
            # [BatchSize, 64, 171]
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=(2,), stride=(2,), padding=1)
            # [BatchSize, 64, 86]
        )

        self.layer5 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=(3,), stride=(1,)),
            # [BatchSize, 64, 84]
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=(2,), stride=(2,))
            # [BatchSize, 64, 42]
        )

        self.fc = nn.Sequential(
            # [BatchSize, 2688]
            nn.Linear(2688, 100),
            # [BatchSize, 100]
            # nn.ReLU(inplace=True),
            # nn.Linear(100, out_channel),
            # nn.Softmax()
            # # [BatchSize, 2]
        )

    def forward(self, x):
        x1 = self.layer1_1(x)
        x2 = self.layer1_2(x)
        x3 = self.layer1_3(x)
        x = torch.cat((x1, x2, x3), dim=2)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # [BatchSize, 64, 42]
        x = x.view(x.size(0), -1)
        # [BatchSize, 2688]
        x = self.fc(x)
        # [BatchSize, 100]
        return x

