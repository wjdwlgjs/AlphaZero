from torch import nn 

class bn_conv_layer(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, padding):
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding, bias = False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class ResUnit(nn.Module):
    def __init__(self, in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1):
        self.network = nn.Sequential(
            bn_conv_layer(in_channels, out_channels, kernel_size, padding),
            bn_conv_layer(in_channels, out_channels, kernel_size, padding)
        )

    def forward(self, x):
        return x + self.network(x)