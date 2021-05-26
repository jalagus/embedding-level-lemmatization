import torch.nn as nn
from torch.nn import functional as F


class LemmatizerNet(nn.Module):
    def __init__(self, embedding_dim=300):
        super(LemmatizerNet, self).__init__()

        self.fc1 = nn.Linear(embedding_dim, 1000)
        self.fc2 = nn.Linear(1000, embedding_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class LinearLemmatizerNet(nn.Module):
    def __init__(self, embedding_dim=300):
        super(LinearLemmatizerNet, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        return self.fc1(x)

class CompressionLemmatizerNet(nn.Module):
    def __init__(self, embedding_dim=300):
        super(CompressionLemmatizerNet, self).__init__()

        self.fc1 = nn.Linear(embedding_dim, 100)
        self.fc2 = nn.Linear(100, embedding_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class ComplexLemmatizerNet(nn.Module):
    def __init__(self, embedding_dim=300):
        super(ComplexLemmatizerNet, self).__init__()

        self.fc1 = nn.Linear(embedding_dim, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, embedding_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
