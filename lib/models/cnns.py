import torch.nn as nn


class PoseCNN(nn.Module):
    def __init__(self, num_joints, hidden_size, output_size):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Conv1d(num_joints, hidden_size, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
        )
        self.res_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
            ) for _ in range(2)
        ])
        self.output_layer = nn.Conv1d(hidden_size, output_size, kernel_size=1, bias=False)

    def forward(self, x):
        """
        Args
            x: [B, Nj, Nd]
        Returns
            [B, Cout, Nd]
        """
        x = self.input_layer(x)
        for l in range(len(self.res_layers)):
            y = self.res_layers[l](x)
            x = x + y
        x = self.output_layer(x)

        return x


class JointCNN(nn.Module):
    def __init__(self, num_joints, hidden_size, output_size):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Conv1d(num_joints, hidden_size, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
        )
        self.res_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1, dilation=1, bias=False),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=2, dilation=2, bias=False),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=4, dilation=4, bias=False),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=8, dilation=8, bias=False),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=16, dilation=16, bias=False),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=32, dilation=32, bias=False),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1, dilation=1, bias=False),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
            ) for _ in range(2)
        ])
        self.output_layer = nn.Conv1d(hidden_size, output_size, kernel_size=1, bias=False)

    def forward(self, x):
        """
        Args
            x: [B, Nj, Nd]
        Returns
            [B, Cout, Nd]
        """
        x = self.input_layer(x)
        for l in range(len(self.res_layers)):
            y = self.res_layers[l](x)
            x = x + y
        x = self.output_layer(x)

        return x
