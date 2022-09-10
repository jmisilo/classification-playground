import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, in_features, out_features, p=0.1):
        super(Net, self).__init__()
        self.inp = nn.Linear(in_features, 8)

        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)
        
        self.out = nn.Linear(8, out_features)

        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = self.silu(self.inp(x))

        x = self.dropout(self.silu(self.fc1(x)))
        x = self.dropout(self.silu(self.fc2(x)))

        x = self.out(x)

        return x if self.out.out_features > 1 else torch.sigmoid(x)

if __name__ == '__main__':
    net = Net(2, 2)
    print(net)